import numpy as np
import matplotlib.pyplot as plt
import time
from collections import deque
from datetime import datetime, timedelta

##DATA STREAM SIMULATION CLASS
class SimulateData:
  def __init__(self, base_value = 100, trend_slope = 0.1, seasonal_amplitude = 20, seasonal_period = 24, noise_level = 5, random_seed = None):
    self.base_value = base_value
    self.trend_slope = trend_slope
    self.seasonal_amplitude = seasonal_amplitude
    self.seasonal_period = seasonal_period
    self.noise_level = noise_level
    self.t = 0
    self.anomaly_prob = 0.05

    if random_seed is not None:
      np.random.seed(random_seed)
  
  ##Trend
  def generate_trend(self, n_periods):
    return np.arange(n_periods) * self.trend_slope
  
  #Seasonality
  def generate_seasonality(self, n_periods):
    return np.sin(np.arange(n_periods) * 2 * np.pi / self.seasonal_period) * self.seasonal_amplitude

  #Noise
  def generate_noise(self, n_periods):
    return np.random.normal(0, self.noise_level, n_periods)
  
  def generate_data_stream(self, n_periods, start_time = None):

    if start_time is None:
      start_time = datetime.now()

    #Generate Components
    trend = self.generate_trend(n_periods)
    seasonality = self.generate_seasonality(n_periods)
    noise = self.generate_noise(n_periods)

    #Combine Components
    values = self.base_value + trend + seasonality + noise
    timestamps = [start_time + timedelta(hours=i) for i in range(n_periods)]

    return timestamps, values

  def generate_next_value(self):

    self.t += 1

    trend = self.trend_slope * self.t
    seasonality = self.seasonal_amplitude * np.sin(self.t * 2 * np.pi / self.seasonal_period)
    noise = np.random.normal(0, self.noise_level)

    value = self.base_value + trend + seasonality + noise

    return value


'''
  Seasonality : We import STL decomposition from statsmodel library
                -> STL decomposes the input into Seasonal + Trend + Residue
                -> By this STL Decomposition library we make sure that our library adapts to seasonal variations

  Drift       : We import ADWIN from river library
              -> It checks whether there is a difference in the distribution of the current point with respect to previous data
              -> By ADWIN we make sure that our algorithm adapts to Drift
'''


'''
   ALGORITHM for Streaming Data Anomaly Detection - 

   EXPONENTIAL MOVING AVERAGE (EMA): Exponential Moving Average focuses more on recent data by assigning more weight to new data points; 
   so, they are weigted by timestamp - most recent has more importance. 
   Further, we simple check if the new record is far from the expected value. 
   The expected value range is computed using the formula Exponential Moving Average + standard deviation * threshold; 
   if it is out of the expected value range, we report as an anomaly.

    Also, the Exponential Moving Average needs a parameter called alpha that determines the importance of the last record, 
    and its value is decreased for the next records. For example, if alpha=0.5: the record-1 has 50% of importance,
    the record-2 has 30% of importance, and so on. Thus, this weighted algorithm enables smooth the expected value.

   WHY EMA IS EFFECTIVE FOR STREAMING DATA ANONALY DETECTION:

   -> Sensitivity to Recent Changes: EMA weighting prioritizes recent values, making it responsive to abrupt shifts, which are often indicative of anomalies.
   -> Low Memory Requirements: Only the latest value and the last EMA value are needed, making EMA memory-efficient, which is ideal for real-time or streaming data.
   -> Noise Reduction: The smoothing factor filters out minor fluctuations, helping to focus on significant deviations, which is essential in distinguishing anomalies from regular noise.
    EMA's ability to adapt quickly while retaining efficiency makes it a strong candidate for anomaly detection in streaming data.

'''
from statsmodels.tsa.seasonal import STL
from river.drift import ADWIN

class AnomalyDetector:

  def __init__(self, window_size = 24, threshold_factor = 2, adapt_rate=0.1):
    self.window_size = window_size
    self.threshold_factor = threshold_factor
    self.adapt_rate = adapt_rate
    self.values = deque(maxlen = self.window_size)
    self.mean = 0
    self.std = 1
    self.X = []
    self.adwin = ADWIN()

  def update_statistic(self, value):
    self.values.append(value)

    if len(self.values) >= self.window_size:
      new_mean = np.mean(self.values)
      new_std = np.std(self.values)

      self.mean = (1 - self.adapt_rate) * self.mean + self.adapt_rate * new_mean
      self.std = (1 - self.adapt_rate) * self.std + self.adapt_rate * new_std

  def detect_anomaly(self, value):

    self.X.append(value)

    if len(self.X) < self.window_size:
      return False

    # Remove the seasonal and trend from the data and obtain the residual
    stl = STL(self.X, period = self.window_size, robust = True)
    res = stl.fit()
    resid = res.resid

    #Update the mean and deviation
    self.update_statistic(resid[-1])

    #CHECK for ANOMALY (EMA)
    dev = self.std * self.threshold_factor
    expected = self.mean

    if expected + dev < resid[-1] or expected - dev > resid[-1]:
      return True

    #check for DRIFT
    if len(self.X) == self.window_size:
      for residual in resid:
        self.adwin.update(residual)
    else :
      self.adwin.update(resid[-1])
      if self.adwin.drift_detected:
        return True

    #If there is no anomaly or drift
    return False

from matplotlib.animation import FuncAnimation
from types import FrameType

class RealTimeVisualization:
      def __init__(self, max_points=200):
        self.max_points = max_points
        self.times = deque(maxlen=max_points)
        self.values = deque(maxlen=max_points)
        self.anomalies_x = deque(maxlen=max_points)
        self.anomalies_y = deque(maxlen=max_points)

        plt.ion()  # Interactive mode
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [], 'b-', label='Data Stream')
        self.anomaly_scatter = self.ax.scatter([], [], color='red', label='Anomalies')
        self.ax.set_title('Real-time Anomaly Detection')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Value')
        self.ax.legend()

      def update(self, value, is_anomaly, frame):
          # Get data for this frame (you'll need to manage this)

          time = frame

          self.times.append(time)
          self.values.append(value)

          if is_anomaly:
              self.anomalies_x.append(time)
              self.anomalies_y.append(value)

          self.line.set_data(list(self.times), list(self.values))
          self.anomaly_scatter.set_offsets(np.c_[list(self.anomalies_x), list(self.anomalies_y)])

          return self.line, self.anomaly_scatter

def start_visualization():
    simulator = SimulateData(random_seed=21)
    detector = AnomalyDetector()
    visualizer = RealTimeVisualization()

    try:
        print("Start ANOMALY DETECTION for Data Stream")
        cnt = 0
        def update(frame):
            nonlocal cnt
            value = simulator.generate_next_value()  # Get value and time
            is_anomaly = detector.detect_anomaly(value)
            cnt += 1
            return visualizer.update(value, is_anomaly, cnt)

        while True:

          ani = FuncAnimation(visualizer.fig,
                              update,
                              interval=100,
                              blit=True,
                              cache_frame_data=False)

          plt.show(block=True)

    except KeyboardInterrupt:
        print("Stop ANOMALY DETECTION")
        plt.close()

if __name__ == "__main__" :
   start_visualization()
