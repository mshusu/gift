# `proc_stream_mode` Calculation Guide

This document summarizes the stream-weight calculation used with:

```bash
python main.py \
  --model_name LGN-or-BPR \
  --dyn_method finetune-globalinfocontent \
  --proc_stream_mode <mode>
```

The selected mode produces a weight $w_i$ for each positive item $i$ in a
training example. `Runner` passes this weight to the model loss.

## Notation

| Symbol | Meaning |
|---|---|
| $t$ | Current snapshot index |
| $S_t$ | Set of items in the current snapshot block |
| $L_t$ | Item list in the current snapshot block |
| $n_t(i)$ | Number of occurrences of item $i$ in $L_t$ |
| $p_t(i)$ | Current block probability, $n_t(i) / \lvert L_t \rvert$ |
| $\alpha$ | `--strefreq_alpha`, interval EMA coefficient |
| $\beta$ | `--strefreq_pro_alpha`, probability/entropy EMA coefficient |
| $\log_2$ | Base-2 logarithm; information is measured in bits |

## Mode Summary

| Mode | State source | Output weight |
|---|---|---|
| `item_set_infoFr` | Item-set interval EMA | Frequency information |
| `item_set_infoPr` | Item-set block counts | Block-probability information |
| `item_list_infoPrEMA` | Item-list probability EMA | Probability information |
| `item_list_infoGlobal` | Cumulative item-list counts | Global probability information |
| `item_list_parEntropyPrEMA` | Probability EMA | Partial entropy |
| `item_list_parEntropyEMA` | Partial-entropy EMA | Partial entropy |
| `item_list_parEntropyGlobal` | Cumulative item-list counts | Global partial entropy |
| `item_both_infoFr` | Item-set interval EMA + item-list probability EMA | Sum of two information values |
| `item_both_infoPr` | Item-set block counts + item-list probability EMA | Sum of two information values |
| `item_both_partialEntropy` | Item-set interval EMA + item-list probability EMA | Sum of two partial entropies |

## Individual Modes

### `item_set_infoFr`

This mode estimates the interval between occurrences of an item in snapshot
item sets.

For an item that occurs at snapshot $t$, the interval is:

$$
g_i = t - t_i^{\mathrm{last}}.
$$

The interval state uses an EMA:

$$
r_i \leftarrow (1-\alpha)r_i + \alpha g_i.
$$

For the first occurrence, the implementation uses a neutral information weight
of $2$. Therefore:

$$
I_i = \log_2\!\bigl(\max(r_i,2)\bigr),
\qquad
p_i = \frac{1}{\max(r_i,2)}.
$$

The returned weight is:

$$
w_i = I_i.
$$

The first occurrence has $w_i = \log_2(2) = 1$ bit.

### `item_set_infoPr`

This mode counts how many processed snapshot blocks contain each item.

Let $c_i$ be the number of processed snapshot blocks containing item $i$,
and let $B$ be the number of processed snapshot blocks:

$$
p_i = \frac{c_i}{B},
\qquad
w_i = -\log_2(p_i) = \log_2\!\left(\frac{B}{c_i}\right).
$$

### `item_list_infoPrEMA`

First calculate the probability of item $i$ in the current item list:

$$
p_t(i) = \frac{n_t(i)}{\lvert L_t \rvert}.
$$

The probabilities are updated with an EMA. For an item that was previously
observed:

$$
\widehat{p}_i \leftarrow
(1-\beta)^{t-t_i^{\mathrm{last}}}\widehat{p}_i
+ \beta p_t(i).
$$

The first observation initializes the state directly with the current block
probability. The returned weight is:

$$
w_i = -\log_2(\widehat{p}_i).
$$

### `item_list_infoGlobal`

This mode uses cumulative item-list counts. Let $C_i$ be the cumulative
occurrences of item $i$, and let $N$ be the cumulative number of item-list
entries:

$$
p_i = \frac{C_i}{N},
\qquad
w_i = -\log_2(p_i) = \log_2\!\left(\frac{N}{C_i}\right).
$$

### `item_list_parEntropyPrEMA`

This mode first computes the probability EMA from
`item_list_infoPrEMA`, then converts it to partial entropy:

$$
w_i = \widehat{p}_i\,\bigl[-\log_2(\widehat{p}_i)\bigr].
$$

### `item_list_parEntropyEMA`

First calculate the current block partial entropy:

$$
h_t(i) = p_t(i)\,\bigl[-\log_2(p_t(i))\bigr].
$$

Then smooth partial entropy directly:

$$
\widehat{h}_i \leftarrow
(1-\beta)^{t-t_i^{\mathrm{last}}}\widehat{h}_i
+ \beta h_t(i).
$$

The first observation initializes the state directly with $h_t(i)$. The
returned weight is:

$$
w_i = \widehat{h}_i.
$$

### `item_list_parEntropyGlobal`

This mode calculates partial entropy from cumulative counts:

$$
p_i = \frac{C_i}{N},
\qquad
w_i = p_i\,[-\log_2(p_i)]
= \frac{C_i}{N}\log_2\!\left(\frac{N}{C_i}\right).
$$

## Combined Modes

### `item_both_infoFr`

This mode uses independent item-set and item-list states:

$$
w_i = w_i^{\mathrm{set,Fr}} + w_i^{\mathrm{list,EMA}}.
$$

Expanded:

$$
w_i = \log_2\!\bigl(\max(r_i,2)\bigr) - \log_2(\widehat{p}_i).
$$

### `item_both_infoPr`

This mode combines snapshot-block probability information with item-list
probability EMA information:

$$
w_i = w_i^{\mathrm{set,Pr}} + w_i^{\mathrm{list,EMA}}.
$$

Expanded:

$$
w_i = \log_2\!\left(\frac{B}{c_i}\right) - \log_2(\widehat{p}_i).
$$

The item-set counts and item-list EMA state are maintained independently.

### `item_both_partialEntropy`

This mode adds the item-set partial entropy and item-list EMA partial entropy.
For $r_i^+ = \max(r_i,2)$:

$$
p_i^{\mathrm{set}} = \frac{1}{r_i^+},
\qquad
I_i^{\mathrm{set}} = \log_2(r_i^+),
$$

$$
w_i = \frac{1}{r_i^+}\log_2(r_i^+)
+ \widehat{p}_i\,[-\log_2(\widehat{p}_i)].
$$

## Effect on LGN and BPR

Both `LGN` and `BPR` use the same `Runner` stream-weight path. The selected
mode changes the weights, not the model architecture.

The weights are applied according to `--gitf_weight_mode`:

$$
\texttt{margin:}\qquad
\mathrm{weighted\_margin} = \mathrm{margin}\times w_i
$$

$$
\texttt{example:}\qquad
\mathrm{weighted\_loss} = \mathrm{loss}\times w_i.
$$
