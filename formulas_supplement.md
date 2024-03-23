# Formulas Supplement 

## Text Features

### Automated Readability Index (ARI)

The Automated Readability Index (ARI) calculates the readability of a text based on the average number of characters per word and the average number of words per sentence. The formula for ARI is:

$$ ARI = 4.71 \times \left( \frac{{\text{characters}}}{{\text{words}}} \right) + 0.5 \times \left( \frac{{\text{words}}}{{\text{sentences}}} \right) - 21.43 $$

### Flesch Readability Index

The Flesch Readability Index measures the ease of understanding of a text based on the average sentence length and the average number of syllables per word. The formula for the Flesch Readability Index is:

$$ FRI = 206.835 - 1.015 \times \left( \frac{{\text{words}}}{{\text{sentences}}} \right) - 84.6 \times \left( \frac{{\text{syllables}}}{{\text{words}}} \right) $$

### Gunning Fog Index

The Gunning Fog Index estimates the readability of a text by considering the number of complex words in relation to the total number of words and sentences. The formula for the Gunning Fog Index is:

$$ GFI = 0.4 \times \left( \frac{{\text{words}}}{{\text{sentences}}} \right) + 100 \times \left( \frac{{\text{complex words}}}{{\text{words}}} \right) $$


### Reading Time

The Reading Time calculates the total milliseconds to read a text assuming 14.69ms per character.

$$ RT = 14.69 * \text{characters} $$

### Words per Sentence

Words per Sentence measures the average words per sentence.

$$ WPS = \frac{\text{words}}{\text{sentences}} $$

### Perplexity 

Perplexity calculates the exponential weighted average of the negative log-likelihoods of a sequence of words $W$.

$$ PPL(W) = exp( -\frac{1}{t} \sum_{i=0}^{t}  log p(w_i | w_{i-1}, w_{i-2}, \dots  ) ) $$ 



## Image Features 

RGB Color Model:

In the RGB color model, colors are represented as combinations of red, green, and blue light. Each pixel in an image is described by three values, corresponding to the intensity of red, green, and blue light required to create the color of that pixel. The intensity values typically range from 0 to 255, where 0 indicates no contribution of that particular color, and 255 indicates full intensity.

This model is additive, meaning that different combinations of red, green, and blue light can produce a wide range of colors. By varying the intensity of each color channel, millions of distinct colors can be represented.



HSV Color Model:
The HSV color model, also known as HSB (Hue, Saturation, Brightness), represents colors in terms of their perceived characteristics: hue, saturation, and value.

Hue (H): It describes the type of color, such as red, blue, or yellow, represented as a degree on a color wheel. Hue essentially determines the dominant wavelength of light.

Saturation (S): Saturation refers to the intensity or purity of the color, ranging from fully saturated (pure color) to unsaturated (gray).

Value (V): Value represents the brightness of the color, ranging from black to white.

The HSV model provides an intuitive way to describe and manipulate colors, especially for tasks such as color selection and adjustment. For instance, by adjusting the saturation, one can make colors more vivid or muted without changing their brightness or hue.

