# Production Scaling Plan — 10,000+ Users

## Current state

The Streamlit + Flask + ngrok setup is an MVP/demo architecture. It is **not** claimed to support 10,000 concurrent users.

## Target architecture

text
                    Web / WhatsApp Users
                              |
                              v
                       CDN / WAF / LB
                              |
                  +-----------+-----------+
                  |           |           |
                API #1      API #2      API #N
                  |           |           |
                  +-----------+-----------+
                              |
                              v
                        Redis / Queue
                              |
                  +-----------+-----------+
                  |           |           |
               ML Worker   ML Worker   ML Worker
                  |           |           |
                  +-----------+-----------+
                       |              |
                       v              v
                Object Storage   PostgreSQL/Supabase
                   (images)       (feedback/metadata)
```

## Scaling decisions

- **Load balancer:** distributes requests across stateless API replicas.
- **Queue:** protects the API from traffic spikes and smooths ML inference load.
- **ML workers:** load the model once per worker and scale horizontally; GPU workers can be added when CPU inference is insufficient.
- **Object storage:** images should not live on application disk.
- **PostgreSQL/Supabase:** stores feedback, metadata, audit information and evaluation records.
- **Rate limiting:** prevents a single client from exhausting inference capacity.
- **Observability:** track latency, queue depth, errors, throughput, CPU/GPU and memory.
- **Autoscaling:** increase/decrease API and inference replicas based on traffic and queue depth.

## Capacity testing before claiming 10k users

1. Define the target as **10,000 concurrent users** or a specific requests/second target.
2. Benchmark the model inference time and memory footprint.
3. Load-test the API with realistic image sizes.
4. Measure p50/p95/p99 latency and error rate.
5. Increase worker replicas until the target SLO is met.
6. Repeat the test after every model or infrastructure change.

## Interview answer

> "The current deployment is an MVP, so I would not claim it handles 10,000 concurrent users. For production scale I would separate the UI from the inference API, put the API behind a load balancer, use Redis for asynchronous image jobs, horizontally scale model workers, move images to object storage, keep metadata in PostgreSQL, and use monitoring plus autoscaling."
