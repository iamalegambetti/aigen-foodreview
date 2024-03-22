# Formulas Supplement 

## Automated Readability Index (ARI)

The Automated Readability Index (ARI) calculates the readability of a text based on the average number of characters per word and the average number of words per sentence. The formula for ARI is:

$$ ARI = 4.71 \times \left( \frac{{\text{characters}}}{{\text{words}}} \right) + 0.5 \times \left( \frac{{\text{words}}}{{\text{sentences}}} \right) - 21.43 $$

## Flesch Readability Index

The Flesch Readability Index measures the ease of understanding of a text based on the average sentence length and the average number of syllables per word. The formula for the Flesch Readability Index is:

$$ FRI = 206.835 - 1.015 \times \left( \frac{{\text{words}}}{{\text{sentences}}} \right) - 84.6 \times \left( \frac{{\text{total syllables}}}{{\text{words}}} \right) $$

## Gunning Fog Index

The Gunning Fog Index estimates the readability of a text by considering the number of complex words in relation to the total number of words and sentences. The formula for the Gunning Fog Index is:

$$ GFI = 0.4 \times \left( \frac{{\text{words}}}{{\text{sentences}}} \right) + 100 \times \left( \frac{{\text{complex words}}}{{\text{words}}} \right) $$


## Reading Time

The Reading Time calculates the total milliseconds to read a text assuming 14.69ms per character.

$$ RT = 14.69 * \text{characters} $$

## Words per Sentence

Words per Sentence measures the average words per sentence.

$$ WPS = \frac{\text{words}}{\text{sentences}} $$

## Perplexity 

Perplexity calculates the exponential weighted average of the negative log-likelihoods of a sequence of words $W$.

$$ PPL(W) = exp( -\frac{1}{t} \sum_{i=0}^{t}  log p(w_i | w_{<i}   ) ) $$ 