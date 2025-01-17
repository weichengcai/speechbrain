#!/usr/bin/env python3
"""Recipe for training a speaker-id system. The template can use used as a
basic example for any signal classification task such as language_id,
emotion recognition, command classification, etc. The proposed task classifies
28 speakers using Mini Librispeech. This task is very easy. In a real
scenario, you need to use datasets with a larger number of speakers such as
the voxceleb one (see recipes/VoxCeleb). Speechbrain has already some built-in
models for signal classifications (see the ECAPA one in
speechbrain.lobes.models.ECAPA_TDNN.py or the xvector in
speechbrain/lobes/models/Xvector.py)

To run this recipe, do the following:
> python train.py train.yaml

To read the code, first scroll to the bottom to see the "main" code.
This gives a high-level overview of what is going on, while the
Brain class definition provides the details of what happens
for each batch during training.

The first time you run it, this script should automatically download
and prepare the Mini Librispeech dataset for computation. Noise and
reverberation are automatically added to each sample from OpenRIR.

Authors
 * Mirco Ravanelli 2021
"""
import os
import sys
import numpy as np
import random
import torch
import torchaudio
import speechbrain as sb
import scipy.io.wavfile as sciwav
from hyperpyyaml import load_hyperpyyaml


# Brain class for speech enhancement training
class SpkIdBrain(sb.Brain):
    def compute_forward(self, batch, stage):
        """Runs all the computation of that transforms the input into the
        output probabilities over the N classes.

        Arguments
        ---------
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        predictions : Tensor
            Tensor that contains the posterior probabilities over the N classes.
        """

        # We first move the batch to the appropriate device.
        batch = batch.to(self.device)
#        print(batch.id, batch.gender_id_encoded)

        # Compute features, embeddings, and predictions
        feats, lens = self.prepare_features(batch.sig, stage)
#        print(feats.shape)
        embeddings = self.modules.embedding_model(feats, lens)
        predictions = self.modules.classifier(embeddings)

        return predictions

    def prepare_features(self, wavs, stage):
        """Prepare the features for computation, including augmentation.

        Arguments
        ---------
        wavs : tuple
            Input signals (tensor) and their relative lengths (tensor).
        stage : sb.Stage
            The current stage of training.
        """
        wavs, lens = wavs

        # Add augmentation if specified. In this version of augmentation, we
        # concatenate the original and the augment batches in a single bigger
        # batch. This is more memory-demanding, but helps to improve the
        # performance. Change it if you run OOM.
        if stage == sb.Stage.TRAIN:
            if hasattr(self.modules, "env_corrupt"):
                wavs_noise = self.modules.env_corrupt(wavs, lens)
                wavs = torch.cat([wavs, wavs_noise], dim=0)
                lens = torch.cat([lens, lens])

            if hasattr(self.modules, "augmentation"):
                wavs = self.modules.augmentation(wavs, lens)

        # Feature extraction and normalization
        feats = self.modules.compute_features(wavs)
        feats = self.modules.mean_var_norm(feats, lens)

        return feats, lens

    def compute_objectives(self, predictions, batch, stage):
        """Computes the loss given the predicted and targeted outputs.

        Arguments
        ---------
        predictions : tensor
            The output tensor from `compute_forward`.
        batch : PaddedBatch
            This batch object contains all the relevant tensors for computation.
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.

        Returns
        -------
        loss : torch.Tensor
            A one-element tensor used for backpropagating the gradient.
        """

        _, lens = batch.sig
        spkid, _ = batch.gender_id_encoded

        # Concatenate labels (due to data augmentation)
        if stage == sb.Stage.TRAIN and hasattr(self.modules, "env_corrupt"):
            spkid = torch.cat([spkid, spkid], dim=0)
            lens = torch.cat([lens, lens])

        # Compute the cost function
#        loss = sb.nnet.losses.nll_loss(predictions, spkid, lens)
#        print(predictions.shape, spkid.shape, lens.shape)
        predictions = predictions.squeeze(-1)
        loss = self.hparams.compute_BCE_cost(predictions, spkid, lens)
    

        self.train_metrics.append(batch.id, torch.sigmoid(predictions), spkid)
        if stage != sb.Stage.TRAIN:
            self.valid_metrics.append(
                batch.id, torch.sigmoid(predictions), spkid
            )

        # Append this batch of losses to the loss metric for easy
#        self.loss_metric.append(
#            batch.id, predictions, spkid, lens, reduction="batch"
#        )

        # Compute classification error at test time
#        if stage != sb.Stage.TRAIN:
#            self.error_metrics.append(batch.id, predictions, spkid, lens)

        return loss

    def on_stage_start(self, stage, epoch=None):
        """Gets called at the beginning of each epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, or sb.Stage.TEST.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        # Set up statistics trackers for this stage
        self.train_metrics = self.hparams.train_stats()

        if stage != sb.Stage.TRAIN:
            self.valid_metrics = self.hparams.test_stats()

#        self.loss_metric = sb.utils.metric_stats.MetricStats(
#            metric=sb.nnet.losses.nll_loss
#        )

        # Set up evaluation-only statistics trackers
#        if stage != sb.Stage.TRAIN:
#            self.error_metrics = self.hparams.error_stats()

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Gets called at the end of an epoch.

        Arguments
        ---------
        stage : sb.Stage
            One of sb.Stage.TRAIN, sb.Stage.VALID, sb.Stage.TEST
        stage_loss : float
            The average loss for all of the data processed in this stage.
        epoch : int
            The currently-starting epoch. This is passed
            `None` during the test stage.
        """

        if stage == sb.Stage.TRAIN:
            self.train_loss = stage_loss
        else:
            summary = self.valid_metrics.summarize(threshold=0.5)

        # Store the train loss until the validation stage.
#        if stage == sb.Stage.TRAIN:
#            self.train_loss = stage_loss

        # Summarize the statistics from the stage for record-keeping.
#        else:
#            stats = {
#                "loss": stage_loss,
#                "error": self.error_metrics.summarize("average"),
#            }

        # At the end of validation...
        if stage == sb.Stage.VALID:

            old_lr, new_lr = self.hparams.lr_annealing(epoch)
            sb.nnet.schedulers.update_learning_rate(self.optimizer, new_lr)

            # The train_logger writes a summary to stdout and to the logfile.
            self.hparams.train_logger.log_stats(
                {"Epoch": epoch, "lr": old_lr},
                train_stats={"loss": self.train_loss},
                valid_stats={"loss": stage_loss, "summary": summary}
            )

            # Save the current checkpoint and delete previous checkpoints,
#            self.checkpointer.save_and_keep_only(meta=stats, min_keys=["error"])
            self.checkpointer.save_and_keep_only(
                meta={"loss": stage_loss, "summary": summary},
                num_to_keep=1,
                min_keys=["loss"],
                name="epoch_{}".format(epoch),
            )

        # We also write statistics about test data to stdout and to the logfile.
        if stage == sb.Stage.TEST:
            self.hparams.train_logger.log_stats(
                {"Epoch loaded": self.hparams.epoch_counter.current},
                test_stats={"loss": stage_loss, "summary": summary},
#                test_stats=stats,
            )


def dataio_prep(hparams):
    """This function prepares the datasets to be used in the brain class.
    It also defines the data processing pipeline through user-defined functions.
    We expect `prepare_mini_librispeech` to have been called before this,
    so that the `train.json`, `valid.json`,  and `valid.json` manifest files
    are available.

    Arguments
    ---------
    hparams : dict
        This dictionary is loaded from the `train.yaml` file, and it includes
        all the hyperparameters needed for dataset construction and loading.

    Returns
    -------
    datasets : dict
        Contains two keys, "train" and "valid" that correspond
        to the appropriate DynamicItemDataset object.
    """

    # Initialization of the label encoder. The label encoder assigns to each
    # of the observed label a unique index (e.g, 'spk01': 0, 'spk02': 1, ..)
    label_encoder = sb.dataio.encoder.CategoricalEncoder()

    snt_len_sample = int(hparams["sample_rate"] * hparams["sentence_len"])

    # Define audio pipeline
    @sb.utils.data_pipeline.takes("wav", "duration")
    @sb.utils.data_pipeline.provides("sig")
    def audio_pipeline(wav, duration):
        """Load the signal, and pass it and its length to the corruption class.
        This is done on the CPU in the `collate_fn`."""

        audio_tensor = sb.dataio.dataio.read_audio(wav)
        while (len(audio_tensor) <= snt_len_sample):
            audio_tensor = audio_tensor.repeat(2)

        if sb.Stage.TRAIN and hparams["random_chunk"]:
            start = random.randint(0, len(audio_tensor) - snt_len_sample - 1)
            stop = start + snt_len_sample
        else:
            start = 0
            stop = len(audio_tensor)
        sig = audio_tensor[start:stop]
        
#        fs, signal = sciwav.read(wav, mmap=True)
#        if signal.shape[0] <= snt_len_sample:
#            signal = np.concatenate([signal] * (math.floor(tlen / signal.shape[0]) + 1), axis=0)
#
#        if hparams["random_chunk"]:
#            start = random.randint(0, len(audio_tensor) - snt_len_sample - 1)
#            stop = start + snt_len_sample
#            signal = signal[start:stop]

#        sig = copy.deepcopy(np.array(signal))
#        sig = torch.from_numpy(sig)

        return sig 

    # Define label pipeline:
    @sb.utils.data_pipeline.takes("gender_id")
    @sb.utils.data_pipeline.provides("gender_id", "gender_id_encoded")
    def label_pipeline(gender_id):
        yield gender_id
        gender_id_encoded = label_encoder.encode_label_torch(gender_id)
        yield gender_id_encoded

    # Define datasets. We also connect the dataset with the data processing
    # functions defined above.
    datasets = {}
    for dataset in ["train", "valid", "test"]:
        datasets[dataset] = sb.dataio.dataset.DynamicItemDataset.from_json(
            json_path=hparams[f"{dataset}_annotation"],
            replacements={"data_root": hparams["data_folder"]},
            dynamic_items=[audio_pipeline, label_pipeline],
            output_keys=["id", "sig", "gender_id_encoded"],
        )

    # Load or compute the label encoder (with multi-GPU DDP support)
    # Please, take a look into the lab_enc_file to see the label to index
    # mapping.
    lab_enc_file = os.path.join(hparams["save_folder"], "label_encoder.txt")
    label_encoder.load_or_create(
        path=lab_enc_file,
        from_didatasets=[datasets["train"]],
        output_key="gender_id",
    )

    return datasets


# Recipe begins!
if __name__ == "__main__":

    # Reading command line arguments.
    hparams_file, run_opts, overrides = sb.parse_arguments(sys.argv[1:])

    # Initialize ddp (useful only for multi-GPU DDP training).
    sb.utils.distributed.ddp_init_group(run_opts)

    # Load hyperparameters file with command-line overrides.
    with open(hparams_file) as fin:
        hparams = load_hyperpyyaml(fin, overrides)

    # Create experiment directory
    sb.create_experiment_directory(
        experiment_directory=hparams["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )

    # Create dataset objects "train", "valid", and "test".
    datasets = dataio_prep(hparams)

    # Initialize the Brain object to prepare for mask training.
    gender_id_brain = SpkIdBrain(
        modules=hparams["modules"],
        opt_class=hparams["opt_class"],
        hparams=hparams,
        run_opts=run_opts,
        checkpointer=hparams["checkpointer"],
    )

    # The `fit()` method iterates the training loop, calling the methods
    # necessary to update the parameters of the model. Since all objects
    # with changing state are managed by the Checkpointer, training can be
    # stopped at any point, and will be resumed on next call.
    gender_id_brain.fit(
        epoch_counter=gender_id_brain.hparams.epoch_counter,
        train_set=datasets["train"],
        valid_set=datasets["valid"],
        train_loader_kwargs=hparams["dataloader_options"],
        valid_loader_kwargs=hparams["dataloader_options"],
    )

    # Load the best checkpoint for evaluation
    test_stats = gender_id_brain.evaluate(
        test_set=datasets["test"],
        min_key="error",
        test_loader_kwargs=hparams["dataloader_options"],
    )
