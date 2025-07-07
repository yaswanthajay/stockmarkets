import schedule
import time
from train import train_model

schedule.every().day.at("18:30").do(train_model)

while True:
    schedule.run_pending()
    time.sleep(60)
