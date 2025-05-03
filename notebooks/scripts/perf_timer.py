import time

class PerfTimer:
  def __init__(self):
    self.start_time = 0
    self.end_time = 0
    
  def start(self):
    self.start_time = time.time()
    
  def stop(self):
    self.end_time = time.time()

  def reset(self):
    self.start_time = 0
    self.end_time = 0
    
  def elapsed_ms(self):
    return (self.end_time - self.start_time) * 1000