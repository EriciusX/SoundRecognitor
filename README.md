# Sound Recognitor

This is a project for EEC 201 25W.

Team name: '**x**'

Team members: Chenghao Xue, Guanyu Mi.

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

## Result

### Test 1

| Test case    | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 |
| ------------ | -- | -- | -- | -- | -- | -- | -- | -- |
| What I heard | s1 | s2 | s7 | s3 | s4 | s5 | s6 | s7 |

Recognition Rate: 87.50%

---

### Test 2

Sampling rate: 12500 Hz
Duration of 256 samples: 20.48 milliseconds

![Waveform](./results/signal_time.png)

Frame size N=512: Maximum energy at 355.68 ms and 366.21 Hz

![Spectrogram](./results/stft.png)

If we extract 80% highest energy, the spectrogram will be 

![Spectrogram](./results/stft_selected.png)

---

### Test 3

![MFCC](./results/mel_filter.png)

---

### Test 4

![MFCC](./results/mfcc.png)

If we extract 80% highest energy, the MFCC will be

![MFCC](./results/mfcc_selected.png)

---

### Test 5

MFCC results for speaker 2 and 10 in 6 and 7 dimensions.

![MFCC_Space](./results/MFCC%20Space.png)

---

### Test 6

Calculate the VQ codewords in test5 and plot them on the same figure.

![MFCC_Space](./results/MFCC%20Space%20with%20VQ.png)
