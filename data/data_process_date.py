from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os
from sklearn.utils import shuffle


class DataProcessor:
    def __init__(self, cfg, multi_file=False):
        cfg.copyAttrib(self)
        self.multi = multi_file
        self.raw_data = None
        self.raw_extra = None  # extra info such as day and time
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.x_c_np = None
        self.x_p_np = None
        self.x_t_np = None
        self.y_np = None
        self.extra_np = None
        self.load_data()

    def load_data(self):
        if self.multi:
            data_files = os.listdir(self.data_path)
            print(f"Loading multiple data files from {self.data_path}")
            raw_data_list = []
            x_c_list = []
            x_p_list = []
            x_t_list = []
            y_list = []
            for d in data_files:
                raw_data_temp = np.load(os.path.join(self.data_path, d))
                raw_data_list.append(raw_data_temp)
                x_closeness, x_period, x_trend, y, _ = self.process_data(raw_data_temp)
                x_c_list.append(x_closeness)
                x_p_list.append(x_period)
                x_t_list.append(x_trend)
                y_list.append(y)
            total_data = np.concatenate(tuple(raw_data_list), axis=0)
            self.x_c_np = np.concatenate(tuple(x_c_list), axis=0)
            self.x_p_np = np.concatenate(tuple(x_p_list), axis=0)
            self.x_t_np = np.concatenate(tuple(x_t_list), axis=0)
            self.y_np = np.concatenate(tuple(y_list), axis=0)
            print(f"total data min: {np.min(total_data)}")
            print(f"total data max: {np.max(total_data)}")
            print(f"total number of data samples: {len(self.y_np)}")
            self.scaler.fit(total_data.reshape(-1, total_data.shape[-1]))

        else:
            self.raw_data = np.load(self.data_path)
            self.get_extra()
            print(f"Loading one numpy data from {self.data_path}")
            print(f"data min: {np.min(self.raw_data)}")
            print(f"data max: {np.max(self.raw_data)}")
            self.x_c_np, self.x_p_np, self.x_t_np, self.y_np, self.extra_np = self.process_data(self.raw_data)
            print(f"total number of data samples: {len(self.y_np)}")
            self.scaler.fit(self.raw_data.reshape(-1, self.raw_data.shape[-1]))

    def get_train_data(self):
        train_num = int(len(self.y_np) * self.train_split)
        val_num = int(len(self.y_np) * self.train_split * 0.1)

        return self.x_c_np[val_num: train_num], self.x_p_np[val_num: train_num], self.x_t_np[val_num: train_num], self.y_np[val_num: train_num], self.extra_np[val_num: train_num]

    def get_val_data(self):
        val_num = int(len(self.y_np) * self.train_split * 0.1)

        return self.x_c_np[0: val_num], self.x_p_np[0: val_num], self.x_t_np[0: val_num], self.y_np[0: val_num], self.extra_np[0: val_num]

    def get_extra(self):
        len_total = self.raw_data.shape[0]
        raw_date_day = np.zeros((len_total, 7))
        raw_date_hour = np.zeros((len_total, self.interval))
        time = np.arange(len_total, dtype=int)
        time_hour = time % self.interval
        time_day = (time // self.interval) % 7
        for i in range(len_total):
            raw_date_hour[i, time_hour[i]] = 1
            raw_date_day[i, time_day[i]] = 1

        self.raw_extra = np.concatenate((raw_date_hour, raw_date_day), axis=1)

    def get_test_data(self):
        train_num = int(len(self.y_np) * self.train_split)

        return self.x_c_np[train_num:], self.x_p_np[train_num:], self.x_t_np[train_num:], self.y_np[train_num:], self.extra_np[train_num:]

    def process_data(self, raw_data):
        y = []
        y_extra = []
        x_closeness = []
        x_period = []
        x_trend = []
        num_sample = len(raw_data) - self.obs_len - self.trend_len

        for i in range(num_sample):
            y.append(raw_data[self.trend_len + self.obs_len + i])
            y_extra.append(self.raw_extra[self.trend_len + self.obs_len + i])
            x_closeness.append(raw_data[self.trend_len + i: self.trend_len + self.obs_len + i])
            x_period.append(raw_data[self.trend_len - self.period_len + i: self.trend_len - self.period_len + self.obs_len + i + 1])
            x_trend.append(raw_data[i: self.obs_len + i + 1])

        y = np.reshape(y, (num_sample, self.grid_c, self.grid_h, self.grid_w))
        x_closeness = np.reshape(x_closeness,
                                 (num_sample, self.obs_len, self.grid_c, self.grid_h, self.grid_w))
        x_period = np.reshape(x_period,
                              (num_sample, self.obs_len + 1, self.grid_c, self.grid_h, self.grid_w))
        x_trend = np.reshape(x_trend,
                             (num_sample, self.obs_len + 1, self.grid_c, self.grid_h, self.grid_w))

        y_extra = np.reshape(y_extra, (num_sample, self.interval+7))
        return x_closeness, x_period, x_trend, y, y_extra


if __name__ == "__main__":
    a = np.load("/home/xuehao/Downloads/VLUC-master/BikeNYC2/flowioK_BikeNYC2_20160701_20160829_30min.npy", allow_pickle=False)
    # print(a)
    b = np.swapaxes(a, 1, 3)
    c = np.swapaxes(b, 2, 3)
    print(np.shape(c))
    np.save("bike_nyc_II.npy", c)
