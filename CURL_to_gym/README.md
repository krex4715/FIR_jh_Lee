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





## Result
I compared the performance of CURL with the performance of SAC with physical state value environment.

### Original CURL Performance in DM_Control
in [CURL(Contrastive Unsupervised Representation Learning for Sample-Efficient Reinforcement Learning)](https://mishalaskin.github.io/curl/), 
Agent was trained in [DM_Control](https://github.com/deepmind/dm_control)


**I check the performance of CURL in 'CartPole' environment of the DM_Control**
|                                  |                                                |                                                |
| :------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|         `Step 5000`                |                 `Step 10000`                 |                   `Step 15000`                 |
| ![step5000](./img/5000_curl.gif)   |         ![step10000](./img/10000_curl.gif)   |         ![step10000](./img/15000_curl.gif)     |
|         `Step 20000`               |             `Step 25000`                     |             `Step 30000`                       |
| ![step15000](./img/20000_curl.gif) |    ![step20000](./img/25000_curl.gif)        |      ![step20000](./img/30000_curl.gif)        |



### Applying to Gym and Comparing the performance with Dynamic State based SAC

**Raw Pixel Observation Based Learning CURL in gym**   
|                                  |                                                |                                                |
| :------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|         `Step 5000`                |                 `Step 10000`                 |                   `Step 15000`                 |
| ![step5000](./img/curl/5000.gif)   |         ![step10000](./img/curl/10000.gif)   |         ![step10000](./img/curl/15000.gif)     |


**Physical Observation Based Learning SAC in gym**   
[details of Observation](https://www.gymlibrary.dev/environments/classic_control/pendulum/)
|                                  |                                                |                                                |
| :------------------------------: | :--------------------------------------------: | :--------------------------------------------: |
|         `Step 5000`                |                 `Step 10000`                 |                   `Step 15000`                 |
| ![step5000](./img/sac/sac5000.gif)   |         ![step10000](./img/sac/sac10000.gif)   |         ![step10000](./img/sac/sac15000.gif)     |

Even CURL is raw Pixel Based Learning Methods,
but we can see the Sample Efficient of CURL :  CURL Algorithm learning was as good as learning with physical information. (in case of CartPole)

