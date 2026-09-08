# Retailrocket View Subsets

These datasets were generated from `Retailrocket-view-dedup.csv`. The source
file was already deduplicated by user-item pair, so no additional
deduplication was performed.

## Datasets

| Dataset | Filter | Interactions | Users | Items |
|---|---|---:|---:|---:|
| `Retailrocket-view-dedup-3core.csv` | Iterative 3-core | 533,855 | 90,301 | 43,368 |
| `Retailrocket-view-dedup-3udeg.csv` | `degree(user) >= 3` | 674,460 | 116,291 | 121,265 |
| `Retailrocket-view-dedup-3ideg.csv` | `degree(item) >= 3` | 1,978,353 | 1,308,056 | 117,264 |

## Filtering

- `3core` repeatedly removes users and items whose current degree is below 3
  until every remaining user and item has degree at least 3.
- `3udeg` is a one-pass user-only filter. It retains every interaction for
  users whose degree in the source dataset is at least 3.
- `3ideg` is a one-pass item-only filter. It retains every interaction for
  items whose degree in the source dataset is at least 3.

## Format

Each file is tab-separated and starts with this header:

```text
user_id\titem_id\ttime
```

Rows remain sorted by timestamp in ascending order. User and item IDs are
re-indexed independently within each output file. Both index spaces are
contiguous and zero-based.

## Validation

| Dataset | User ID range | Item ID range | Minimum user degree | Minimum item degree |
|---|---:|---:|---:|---:|
| `Retailrocket-view-dedup-3core.csv` | 0-90,300 | 0-43,367 | 3 | 3 |
| `Retailrocket-view-dedup-3udeg.csv` | 0-116,290 | 0-121,264 | 3 | 1 |
| `Retailrocket-view-dedup-3ideg.csv` | 0-1,308,055 | 0-117,263 | 1 | 3 |

All outputs were verified for the expected row counts, unique user-item pairs,
timestamp ordering, contiguous index ranges, and applicable degree thresholds.
