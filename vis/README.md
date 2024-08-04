

## Runtime

```
python runtime.py --benchmark 6200.trip-booking --visualization violin -p gcp/batch-size-30-reps-6 aws/batch-size-30-reps-6 azure/batch-size-30-reps-6 -m 128 -e burst
```

## Scalability

```
python scalability.py --benchmark 6200.trip-booking -p gcp/batch-size-30-reps-6 aws/batch-size-30-reps-6 azure/batch-size-30-reps-6 -m 128 -e burst
```

## Cold Starts

```
python cold_starts.py -p gcp aws azure -e burst
```

## Overhead

```
python overhead.py -b  6200.trip-booking -e burst -m 128 -p gcp/batch-size-30-reps-6 aws/batch-size-30-reps-6 azure/batch-size-30-reps-6
```
```
```
```
