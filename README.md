This is the code for a paper titiled "Wireless Recording and Autoencoder Denoising of Intestinal Activity in Freely Moving Rats"

# WirelessIntestine

In this repository, we have 3 main components used in our paper.

`kicad/` : PCB and BOM of a custom-made board  
`nrf/` : a custom-made firmware for microcontroller (nrf51822)  
`analysis/model_train.py`: codes for training the denoising autoencoder  

## Building a firmware for nrf51822

Building and flashing firmware were described in the Makefile routines.
Connect the debugger to the microcontroller via JTAG, then build and flash the program using:

```
cd nrf/
make . -B ./_build
make flash_softdevice
```

## Detailed description of methods

Below, detailed description of analysis methods utilized in paper.

## Locomotion index

Locomotion index was calculated by the video of behaving rats in their homecage, recorded at a resolution of 960 by 540 pixels squared at 30 frames per second. As an initial preprocessing step for analyzing rats’ locomotion, video frames were converted to grayscale, with frames extracted every 180 frames (downsampled). The differential between successive frames was employed to quantify the magnitude of locomotion at each temporal point. In calculating this specific differential, the variance between the luminance values in each pixel, ranging from 0 to 255, was computed for every frame and aggregated across the entire frame. These aggregated values were designated as the locomotion levels and interpreted as an index of the rat's activity at specific intervals.
Locomotion levels were measured throughout the entire recording period. Home cage provided a confined space, restricting the variety of locomotion. Given this limitation, we estimated that their movements under these experimental conditions could be classified into two distinct states: moving (behaving) and non-moving (immobile). A Gaussian mixture model (GMM) was used to classify these states and determine the threshold for the locomotion index.

## Calculation of alpha in power-law noises

Alpha (α) is a parameter of power-law noise that defines the slope of the spectrum. Specifically, α is determined by the following relationship:
Where f is the frequency and S(f) is the power spectral density (PSD) of the power-law noise,
S(f) ∝ f ^(-α)
This equation indicates that a higher α corresponds to a steeper spectral slope, while a lower α results in a flatter spectrum in the Fourier-transformed waveforms. 

For the analysis of intestinal movement signals, a Fourier transform was applied to the acquired signals and represented on a double-logarithmic graph (Figure 1C, 2D). The signal comprising 180,000 data points was limited within the frequency range up to 5 Hz resulting in 18,000 points, meticulously smoothed employing a third-order Savitzky-Golay filter with a window size of 51, as executed via the Python Scikit-learn library. For the analysis across both logarithmic graphs, the smoothed signal was resampled at uniformly distributed intervals with a scaling factor of 100.02 in the logarithmic domain to circumvent bias in the subsequent fitting procedure. This precaution addresses the intrinsic characteristic of data sampled in linear space, which predisposes it to contain more data points in high-frequency regions as opposed to the low-frequency areas, resulting in their disproportionate representation when analyzed on a logarithmic scale. The resampled signal to mitigate this biased representation was depicted on a logarithmic scale spanning from 5 × 10⁻⁴ to 5 Hz, and thereafter, it was meticulously fitted with a linear function utilizing the least-squares methodology. This linear fit was employed to estimate a robust baseline for evaluating the strength of intestinal movement and calculate the slope of baseline, interpreted as the alpha.

## Definition of Intestinal Events

The amplitude of intestinal movements was segmented into 10.24-second bins. These amplitudes were then classified into two states—containing or lacking intestinal activity—using a Gaussian mixture model (the number of components is set to 2). Intestinal events were defined as periods in which intestinal activity was present.

## Autoencoder construction and training

### Training datasets

The training data comprised scalograms derived from amplitude-modulated waves, representing pure intestinal movements, amalgamated with pink noise to simulate motion-induced disruptions. The pure signal was modeled as the product of a modulation sine wave (frequency: 0.01-0.2 Hz) and a carrier wave (frequency: 0.5-0.7 Hz), with its intensity and frequency, varied across training samples to introduce diversity. Pink noise, inversely proportional to frequency, was incorporated to emulate movement artifacts, with a 15-fold difference in noise intensity between high- and low-noise conditions. The model was trained with 100 artificial 2000-second waves. The model was engineered to receive the high-noise version of signals as input and produce the low-noise version as output. The dataset was split into training and validation groups at an 8:2 ratio.

### Data preprocessing

For preprocessing, signals were converted into scalograms utilizing a short-term Fourier transform (STFT) with a 10.24-second Hann window, providing a frequency resolution of 0.1 Hz. These scalograms, restricted to the 0.4–6.5 Hz range, were segmented into 655-second segments and resized into 64 × 64 matrices for input into the autoencoder.

### Model construction

The autoencoder architecture comprises three parts: the encoder, bottleneck, and decoder. The encoder first processes the input matrices through a 2D convolutional layer with 32 filters of size 3 × 3, followed by a 2 × 2 max-pooling layer, reducing the matrices to 32 × 32 × 32. A second convolutional layer with identical filter dimensions and another max-pooling layer further reduces the size to 64 × 16 × 16. A 2D convolutional operation is performed in the bottleneck layer, increasing the depth to 128 while maintaining the spatial dimensions.
In the decoder, two upsampling layers, each followed by a convolutional layer, sequentially doubled the spatial dimensions and reduced the depth, ultimately restoring the matrices to 32 × 64 × 64. The final convolutional layer employed a linear activation function to conform the output to the input dimensions.

### Training

The model was trained to utilize the Adam optimizer with a learning rate of 0.001 and mean squared error (MSE) as the loss function. The training was confined to 100 epochs, with the learning rate reduced by a factor of 0.1 after five epochs without improvement. If no improvement was observed after ten epochs, training was stopped early. All routines were implemented using Python TensorFlow and Keras libraries.