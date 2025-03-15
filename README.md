# Sound Recognitor

This is a project for EEC 201 25W.

Team name: '**x**'

Team members: **Chenghao Xue**, **Guanyu Mi**.

## File Description

### Files

| Name              | Description                                        |
| ----------------- | -------------------------------------------------- |
| soundrecognitor.m | Main program, used for training and testing data   |
| visualization.m   | For visualization programs                         |
| mfcc.m            | Function to generate MFCC features                 |
| vq_lbg.m          | Function to get VQ codeword based on LBG algorithm |

### Function parameter description

#### mfcc

- Input:
  - inputData: Audio file or Signal
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

## Project Tasks

---

You can use [visualization.m](./visualization.m) to show the results from Test 1 to Test 6.

```shell
matlab -batch "run('visualization.m')"
```

### Test 1: Human recognition performance

The goal of this test is to evaluate human performance in recognizing speakers based on voice recordings of the word "zero." This serves as a benchmark for later comparison with machine learning models.

- Played each sound file in the TRAIN folder.
- Played each sound file in the TEST folder in a random order.
- Attempted to identify the speaker manually without checking the ground truth.
- Recorded the recognition rate as a benchmark for comparison with automated methods.

| Test case    | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| ------------ | -- | -- | -- | -- | -- | -- | -- | -- |
| What I heard | s1 | s2 | s7 | s3 | s4 | s5 | s6 | s7 |

Recognition Rate: 87.50%

+ I actually think s3 in the training set hears different from s3 in the test set.

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
   - Identified the region in the plot that contains most of the energy, both in:

Frame size N=128: Maximum energy at 605.44 ms and 781.25 Hz
Frame size N=256: Maximum energy at 591.60 ms and 732.42 Hz
Frame size N=512: Maximum energy at 574.56 ms and 756.84 Hz

![Spectrogram](./results/stft.png)

Since N=512 has the highest frequency resolution, it is the best choice for the frame size.

---

### TEST 3: Mel-Spaced Filter Bank and Spectrum Analysis

The goal of this test is to analyze the mel-spaced filter bank responses and compare them with theoretical triangular filter shapes.

1. **Mel-Spaced Filter Bank Responses:**
   - Used `melfb.m` given to generate the mel filter bank.
   - Plotted the mel-spaced filter bank responses.
   - Compared the plotted responses with the expected theoretical triangular filter shapes.

![MFCC](./results/mel_filter.png)

2. **Spectrum Before and After Mel Wrapping:**
   - Computed and plotted the spectrum of a speech file:

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

### Test 5

MFCC results for speaker 2 and 10 in 6 and 7 dimensions.

![MFCC_Space](./results/MFCC%20Space.png)

---

### Test 6

Calculate the VQ codewords in test5 and plot them on the same figure.

![MFCC_Space](./results/MFCC%20Space%20with%20VQ.png)

---

### Test 7

Prediction results.

![Test7](./results/Test7.png)

---

### Test 8

Predict the result after notch filter.

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

### Test 10a

Question 1:

![Test10a_0](./results/Test10a_0.png)
![Test10a_12](./results/Test10a_12.png)

Question 2:

![Test10a_combine](./results/Test10a_combine.png)

---

### Test 10b

Question 3:

![Test10b_5](./results/Test10b_5.png)
![Test10b_11](./results/Test10b_11.png)

Question 4:

![Test10b_combine](./results/Test10b_combine.png)
