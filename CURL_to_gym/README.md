# CURL + Gym

I used the [CURL](https://github.com/MishaLaskin/curl) algorithm to learn the agent through the Pixel value in the [gym](https://www.gymlibrary.dev/) environment.



[PixelObservationWrapper](https://www.gymlibrary.dev/api/wrappers/) was used to receive Raw Pixel as observation, not Physical State Value.
```python
env = PixelObservationWrapper(gym.make(args.domain_name))
```



## How to run
### Pendulum Environment
```bash
bash ./script/run_pendulum.sh
```





