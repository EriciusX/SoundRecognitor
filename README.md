# Sound Recognitor

This is a project for EEC 201 25W.

Team name: '**x**'.

Team members (contribution): **Chenghao Xue** (50%), **Guanyu Mi** (50%).

## File Description

### Files

| Name              | Description                                        |
| ----------------- | -------------------------------------------------- |
| soundrecognitor.m | Main program, used for training and testing data   |
| visualization.m   | For visualization programs                         |
| test9.m           | Specialized program for test 9                     |
| test10a.m         | Specialized program for test 10a                   |
| test10b.m         | Specialized program for test 10b                   |
| mfcc.m            | Function to generate MFCC features                 |
| vq_lbg.m          | Function to get VQ codeword based on LBG algorithm |
| autoTrimSilence.m | Function to remove non-vocal parts of audio        |

### Function parameter description

#### mfcc

- Input:
  - y: Signal
  - Fs: Sample Rate
  - N: Frame size (default: 512)
  - num_mel_filters: Number of Mel filters (default: 20)
  - mfcc_coeff: Number of MFCC coefficients (default: 13)
  - select_coef: Selector for frame filtering based on power (default: 1)
- Output:
  - mfcc_features: Matrix of MFCC features for the selected frames

#### vq_lbg

- Input:
  - mfcc    : MFCC matrix
  - M       : The desired number of codewords in the final codebook (default: 8)
  - epsilon : Splitting parameter (default: 0.01)
  - tol     : Iteration stopping threshold (default: 1e-3)
- Output:
  - codebook: An *M x d* matrix, each row is one final codeword

#### autoTrimSilence

- Input:
  - audioFile       : Path to the input audio file
  - frameSize       : Number of samples in each frame
  - thresholdFactor : The fraction of the maximum energy used as a threshold (default: 0.01)
  - overlapRatio    : Overlap ratio for consecutive frames (default: 2/3)
- Output:
  - trimmedSignal   : Audio signal after removing silent parts from the beginning and the end

## Project Tasks

---

You can use [visualization.m](./visualization.m) to show the results from Test 1 to Test 6.

```shell
matlab -batch "run('visualization.m')"
```

### Speech Preprocessing

#### Test 1: Human recognition performance

The goal of this test is to evaluate human performance in recognizing speakers based on voice recordings of the word "zero." This serves as a benchmark for later comparison with machine learning models.

- Played each sound file in the TRAIN folder.
- Played each sound file in the TEST folder in a random order.
- Attempted to identify the speaker manually without checking the ground truth.
- Recorded the recognition rate as a benchmark for comparison with automated methods.

| Test case    | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| ------------ | -- | -- | -- | -- | -- | -- | -- | -- |
| What I heard | s1 | s2 | s7 | s3 | s4 | s5 | s6 | s7 |

Recognition Rate: 87.50%

+ I think s3 in the training set hears different from s3 in the test set.

---

### TEST 2: Analyzing Speech Signal

The purpose of this test is to analyze the speech signal by  visualizing the signal in the time domain. The `autoTrimSilence.m` function is also used to trim silence from the signal data, focusing on the voice  segment.

1. **Play Sound in MATLAB:**

   - Used the `sound` function in MATLAB to play the audio file.
2. **Time Duration Calculation:**

   - Given the sampling rate, calculated how many milliseconds of speech are contained in a block of 256 samples using the formula:
     $$
     \text{Duration (ms)} = \frac{256}{\text{Sampling Rate}} \times 1000
     $$

   Sampling rate: 12500 Hz
   Duration of 256 samples: 20.48 milliseconds
3. **Signal Visualization:**

   - Plotted the normalized signal in the time domain to observe its characteristics.
   - We think the silent segment may affect the classification result, so we used the [autoTrimSilence.m](autoTrimSilence.m) function and 0.01 as the threshold ratio to remove silence from the signal and focus on the voice segment.
   - The untrimmed and trimmed signals were plotted together for comparison.

![Waveform](./results/signal_time.png)

The trimmed signal removes the silent portions, providing a clearer view of the voice segment, which helps in focusing the analysis on the actual speech data.

4. **STFT:**
   - Tried different frame sizes $ N $ (128, 256, and 512) with frame increment $ M \approx \frac{N}{3} $.
   - Identified the region in the plot that contains most of the energy.
   - Also plot the power spectrum of the trimmed signal.

Frame size N=128: Maximum energy at 605.44 ms and 781.25 Hz
Frame size N=256: Maximum energy at 591.60 ms and 732.42 Hz
Frame size N=512: Maximum energy at 574.56 ms and 756.84 Hz

![STFT](./results/stft.png)

Since N=512 has the highest frequency resolution, it is the best choice for the frame size.

---

### TEST 3: Mel-Spaced Filter Bank and Spectrum Analysis

The goal of this test is to analyze the mel-spaced filter bank responses and compare them with theoretical triangular filter shapes.

1. **Mel-Spaced Filter Bank Responses:**
   - Used [melfb.m](melfb.m) given to generate the mel filter bank.
   - Plotted the mel-spaced filter bank responses.
   - Compared the plotted responses with the expected theoretical triangular filter shapes.

![MFCC](./results/mel_filter.png)

2. **Spectrum Before and After Mel Wrapping:**
   - Computed and plotted the spectrum after mel wrapping.
   - Compared the mel-wrapped spectrum with the original spectrum.

![Spectrum](./results/mel_compare.png)

3. **Effect of Mel Wrapping:**
   - The mel-wrapped spectrum compresses higher frequencies while maintaining resolution in the lower frequency range, which reflects human auditory perception.

---

### Test 4: Cepstrum Calculation and MFCC Function Integration

1. **Cepstrum Calculation:**
   - Applied the Discrete Cosine Transform (DCT) to the log mel spectrum to compute the cepstral coefficients.
   - Kept the first few coefficients and remove the first coefficient.

![MFCC](./results/mfcc.png)

2. **MFCC Function Integration:**
   - Combined all steps into a single MATLAB function ([mfcc.m](mfcc.m)) to generate MFCC features from an input speech signal.:
     - Preprocessing (silence removal, normalization)
     - Short-Time Fourier Transform (STFT)
     - Mel filter bank processing
     - Logarithmic scaling
     - Cepstrum calculation using DCT
   - Output: MFCC features for each frame of the input speech signal

---

### Vector Quantization

#### Test 5

We obtain MFCC results for speaker 2 and 10 in the 6th and 7th dimensions.

As shown below, the clustering effect of the trimmed audio is better than that of the original audio.

![MFCC_Space](./results/test5.png)

---

#### Test 6

Then, calculate the VQ codewords in test5 and plot them on the same figure.

![MFCC_Space](./results/test6.png)

---

### Full Test and Demonstration

#### Test 7

Record the prediction results in the dataset **GivenSpeech_Data**.

The autoTrimSilence function is used to preprocess the audio data in both the training and prediction stages.

![Test7](./results/Test7.png)

---

#### Test 8

Then, Add a notch filter to test the system's robustness.

The parameters of  IIR notch filter are:

- Center frequency ($f_0$) = 1500
- Quality factor ($Q$) = 30
- Pole radius ($R$): 1

![Test8](./results/Test8.png)

---

### Test 9

You can get the results of Test 9 by running the following command:

```shell
matlab -batch "run('test_9.m')"
```

then you will first get speakers used in the training set, including 10 speakers randomly chosen from 2024StudentAudioRecording , and then the true and predicted speakers for each test file, and finally the overall recognition rate.

```shell
Speaker used in training set:
2024 student s19
2024 student s10
2024 student s17
2024 student s6
2024 student s2
2024 student s11
2024 student s15
2024 student s13
2024 student s8
2024 student s14
non-student s1
non-student s2
non-student s3
non-student s4
non-student s5
non-student s6
non-student s7
non-student s8
non-student s9
non-student s10
non-student s11

True Speaker: 2024 student s19, Predicted Speaker: 2024 student s19
True Speaker: 2024 student s10, Predicted Speaker: 2024 student s10
True Speaker: 2024 student s17, Predicted Speaker: 2024 student s17
True Speaker: 2024 student s6, Predicted Speaker: 2024 student s6
True Speaker: 2024 student s2, Predicted Speaker: 2024 student s2
True Speaker: 2024 student s11, Predicted Speaker: 2024 student s11
True Speaker: 2024 student s15, Predicted Speaker: 2024 student s15
True Speaker: 2024 student s13, Predicted Speaker: 2024 student s13
True Speaker: 2024 student s8, Predicted Speaker: 2024 student s8
True Speaker: 2024 student s14, Predicted Speaker: 2024 student s14
True Speaker: non-student s1, Predicted Speaker: non-student s1
True Speaker: non-student s2, Predicted Speaker: non-student s2
True Speaker: non-student s3, Predicted Speaker: non-student s7
True Speaker: non-student s4, Predicted Speaker: non-student s4
True Speaker: non-student s5, Predicted Speaker: non-student s5
True Speaker: non-student s6, Predicted Speaker: non-student s6
True Speaker: non-student s7, Predicted Speaker: non-student s7
True Speaker: non-student s8, Predicted Speaker: non-student s8
Overall Recognition Rate: 94.44%
```

---

#### Test 10a

Dataset: **2024StudentAudioRecording**

1. Question 1: If we use "twelve" to identify speakers, what is the accuracy versus the system that uses "zero"?

   Result: "twelve" and "zero" both have the same accuracy of 100%.

   ![Test10a_0](./results/Test10a_0.png)
   ![Test10a_12](./results/Test10a_12.png)

   ---
2. Question 2: If we use train a whole system that tries to identify a) which speaker, and b) whether the speed is "zero" or "twelve", how accurate is your system?

   Method: We trained a combined VQ codebook, which includes the codebooks for "zero" and "twelve." For the input test set, we compute the distance to each codebook, and the one with the shorter distance corresponds to the respective speech type.

   Result: The Prediction accuracy of both a) and b) is 100%.

   ![Test10a_combine](./results/Test10a_combine.png)

---

#### Test 10b

Dataset: **EEC201AudioRecordings**

1. Question 3: If we use "eleven" to identify speakers, what is the accuracy versus the system that uses "five"?

   Result: "five" has a lower accuracy (91.30%), "eleven" has an accuracy of 100%. After manual identification, the timbres of speaker 18 and speaker 13 are very similar, and we believe this is the reason for the prediction error.

   ![Test10b_5](./results/Test10b_5.png)
   ![Test10b_11](./results/Test10b_11.png)

   ---
2. Question 4: How well do they compare against test in 10a using zero/twelve?

   Result: Compared to Question 2 in 10a, its speaker recognition accuracy has decreased. The reason also appears in Question 3, where speaker 18's pronunciation of "five" is mispredicted as speaker 13.

   ![Test10b_combine](./results/Test10b_combine.png)
