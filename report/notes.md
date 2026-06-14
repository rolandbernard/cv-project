Report of 5 to max 10 pages.
* Title: 3D Human Pose Tracking From Multi-View 2D Detections
* Abstract (~150 words)
* Introduction
  * Motivation: Applications in healthcare (e.g., remote monitoring, gait analysis, fall detection), robotics (e.g., understanding surroundings, avoid humans coming close to dangerous machinery), crowd control, virtual reality, and entertainment (e.g., animation for virtual effects, use of body as input device for gaming or virtual avatars).
  * Objective: Investigating Convolutional Recurrent Neural Networks (ConvGRUs) for joint spatial and temporal modeling.
    * Comparison of CNN with stacked frame input vs. ConvGRU models.
    * Short term horizon, 1 hour ahead during training, extended to 4 hours ahead for evaluation.
  * Outline the report structure.
* Dataset Description
  * The CloudCast dataset: 70,080 satellite images (2017–2018) at 15-minute intervals.
  * Dataset derived from satellite images originating from Meteosat Second Generation (MSG) satellites.
  * Dataset already preprocessed and classified into cloud types using an segmentation algorithm by the dataset collectors.
  * A small part of the images are always missing values. These have a separate class in the dataset but is mapped to no clouds in this project. No further preprocessing has been performed as part of this project.
  * Data characteristics: 11 cloud categories, 128×128 pixel resolution, one cloud type per pixel, fixed geographic area.
  * Pre-generated data splits: Chronological split resulting in 52,416 training and 17,664 testing instances. Chronological split is suboptimal because due to seasonal variations the distribution is slightly different. This project still uses the provided split for comparability with other works using the same dataset.
  * Data exploration results:
    * Class distribution over the 11 cloud types. (Also serves to show which types exist.)
    * Persistence of patterns: High similarity between consecutive frames suggests a strong baseline for copying last-frame for prediction.
    * Time-of-day and season dependency: Distribution of cloud types varies by time of the day and across the year.
    * Location dependency: Distribution of cloud types varies across locations, i.e., pixels in the images.
* Background (reference the lecture notes)
  * Convolutional Neural Networks
  * ConvGRU: Explain standard GRU cell and how to extend it to ConvGRU.
  * Autoregressive Models: Explain how the prediction of model can be used to feed to the next invocation to predict more. Used during evaluation even though not explicitly trained this way.
  * Regularization: Explain Weight decay, normalization, dropout, and early stopping. (Techniques used in the project.)
* Methodology 
  * Model Architecture
    * Shared Components.
      * Pixel-wise embedding. Learnable embedding for each cloud type.
      * Using Group normalization to ensure good gradient flow. Chosen over batch normalization due to small batch sizes and stability with recurrent networks.
      * Downsampling using max pooling.
      * Upsampling using Bilinear interpolation.
      * One final convolution without activation function to predict per-pixel class logits.
      * Dropout only used at deeper layers to avoid too much noise during training.
    * Stacked Frames CNN
      * U-net like architecture using concatenated frames to predict the next sequence directly without explicit recurrence.
        * Encoder: ConvBlocks with increasing channel and downsampling.
        * Bottleneck: A single ConvBlock.
        * Decoder: Upsampling and skip-connections from the encoder concatenated at each layer.
      * High number of input and output channels.
      * 25.2M trainable parameters.
    * ConvGRU based Model
      * Recurrent architecture designed to capture temporal dependencies more explicitly than static CNNs.
        * Encoder: Iteratively processes the past 8 frames. ConvGRUBlocks with increasing channels and downsampling.
        * Forecaster: Generates future steps by passing the last hidden state. Reverse direction to the encoder, ConvGRUBlocks and upsampling.
      * Using an Encoder-Forecaster structure.
      * Prediction in the forecaster based purely on rollout in the latent space without feedback.
        * For evaluation, we also test with feedback.
      * 41.6M trainable parameters.
    * Auxiliary Time and Location Embeddings
      * Enhancing the ConvGRU model with learnable features representing the observation timestamp and geographic coordinates.
      * Motivated by the observed time and location dependence of cloud types during data exploration.
      * Sin/Cos time embeddings with periods of a day and a year.
      * Learnable location embeddings at every depth of the encoder and forecaster.
      * Injected using concatenation before each layer.
      * 42.4M trainable parameters.
  * Training Procedure
    * Hyperparameter tuning: Use of Bayesian optimization via the Optuna Python library and Hyperband pruning to optimize learning rate, weight decay, dropout, hidden dimensions, and depth. Using 10% of the dataset and with minimizing loss after 5 epochs.
    * Loss function: Mean pixel-wise multi-class cross-entropy loss.
    * Optimization: Adam optimizer with tuned learning rates
    * Learning rate schedule: Metric based learning rate schedule. Halving learning rate if validation loss does not improve for three epochs.
    * Early stopping: To avoid overfitting and wasted computation, training is stopped after a maximum of 50 epochs or when the validation loss does not improve for 6 epochs.
    * Training on predicting the next four frames given the last 8. Forecast duration kept small due to limited compute resources.
    * Batch size 32 for CNN. Batch size of 8 for ConvGRU based model due to larger memory requirements.
    * Validation strategy: Select 8 equally spaced windows of one weak duration from the dataset. Gap around selected windows to avoid information leakage.
      * Represents approximately a 90%-10% train-validation split.
      * Avoids problem of validation set only covering one season, as would be the case for a purely chronological split.
  * Evaluation Metrics
    * Mean pixel-wise accuracy and Cross-Entropy Loss on the held-out test set.
    * Additionally split by forecast horizon.
    * Comparison baseline: "Copy Model" (predicting the last observed frame for all future steps).
* Results and Discussion
  * Quantitative Results
    * Training/validation loss curves showing convergence for CNN, ConvGRU, and ConvGRU+Embed models.
      * CNN was stopped due to reaching the maximum number of epochs. Still shows very slow improvements and no overfitting.
      * Both ConvGRU based models converted much faster and were stopped by early stopping due to the start of mild overfitting.
    * Comparative test performance: Accuracy vs. forecast horizon.
      * All deep learning models well above the copy baseline. ConvGRU and ConvGRU+Embed provide small but consistent improvement over CNN.
    * When using autoregressive feedback longer horizon forecasts perform better than using purely latent space dynamics during rollout.
  * Qualitative Results
    * Visual analysis of predicted cloud maps vs. ground truth.
    * All models exhibit similar results. Predictions get progressively more blurred as the prediction horizon is increased.
    * Maintains temporal consistency, i.e., no flickering or sudden changes.
* Conclusion and Outlook
  * Summary of the performed work and obtained results
    * Conclude that minor performance improvement of ConvGRU likely does not justify increased compute need.
  * Possible future improvements
    * Natively train with fully autoregressive feedback. Given that during evaluation this strategy worked better, even if not explicitly trained for it.
    * Scale up the CNN or train it for longer to see if it can surpass the ConvGRU.
    * Using different, more advanced RNN layers, e.g., TrajGRU or LSTA.
    * Different loss function such as SSIM. Possibly also as evaluation metric.
      * However, for weather forecasting accuracy at individual locations seems more important than sharp results.
    * More extensive hyperparameter tuning.
    * Use additional information sources, such as multispectral images and temperature maps.
* References
