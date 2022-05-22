import numpy as np
import pandas as pd
import random

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

matplotlib.rcParams['font.sans-serif'] = ['KaiTi']
matplotlib.rcParams['axes.unicode_minus'] = False


def readData():
    label = np.asarray(pd.read_csv('data/label.txt', header=None)).T[0]
    features = np.asarray(pd.read_csv('data/features.txt', header=None))
    data = {"label": label, "features": features}
    return data


def eucliDist(A, B):  # 欧氏距离
    return np.linalg.norm(A - B)


def mahalanobisDist(A, B, data):  # 马氏距离
    invD = np.linalg.inv(np.cov(data.T))
    print(np.dot(np.dot(A - B, invD), (A - B).T))
    return np.sqrt(np.dot(np.dot(A - B.T, invD), (A - B)))


def cosDist(A, B):  # 余弦距离
    return sum(A * B) / (sum(A ** 2) * sum(B ** 2)) ** 0.5


def L1Dist(A, B):  # L1距离
    return np.sum(np.abs(A - B))


def L2Dist(A, B):  # L2距离
    return np.sum(np.square(A - B))


class K_Means:
    def __init__(self, K, data, method):
        self.K = K
        self.data = data
        self.N, self.M = data["features"].shape
        self.method = method

        # 随机取k个样本做中心集
        ran_list = random.sample(range(0, self.N - 1), K)
        # TODO：仅展示不同距离计算方法时使用这行↓
        # ran_list = [5, 16, 27, 38, 49, 60, 71, 82, 93, 104, 115, 126, 137, 148, 159]
        self.center_set = np.zeros([self.K, self.M])
        for i in range(self.K):
            self.center_set[i] = self.data["features"][ran_list[i]]
        print("初始中心集：", ran_list)

        self.distance_set = np.zeros([self.N, K])
        self.predict = np.array([self.N, 1])
        self.epochs = 1
        self.accuracy_list = []
        self.loss_list = []
        self.predict_list = []

        self.train()
        self.visualize()

    def calDist(self):
        if self.method == 1:  # 欧氏距离
            for i in range(self.N):
                for j in range(self.K):
                    self.distance_set[i][j] = eucliDist(self.data["features"][i], self.center_set[j])
        elif self.method == 2:  # 马氏距离
            for i in range(self.N):
                for j in range(self.K):
                    self.distance_set[i][j] = mahalanobisDist(self.data["features"][i], self.center_set[j],
                                                              self.data["features"])
        elif self.method == 3:  # 余弦距离
            for i in range(self.N):
                for j in range(self.K):
                    self.distance_set[i][j] = cosDist(self.data["features"][i], self.center_set[j])
        elif self.method == 4:  # l1距离
            for i in range(self.N):
                for j in range(self.K):
                    self.distance_set[i][j] = L1Dist(self.data["features"][i], self.center_set[j])
        elif self.method == 5:  # l2距离
            for i in range(self.N):
                for j in range(self.K):
                    self.distance_set[i][j] = L2Dist(self.data["features"][i], self.center_set[j])
        # print("距离集：")
        # print(self.distance_set)

    def classify(self):
        self.predict = np.argmin(self.distance_set, axis=1)
        first_present = {}  # 记录簇名第一次出现的位置
        for i in range(self.K):
            for j in range(self.N):
                if self.predict[j] == i:
                    first_present[i] = j
                    break
        ordered = list(dict(sorted(first_present.items(), key=lambda x: x[1], reverse=False)).keys())  # 记录簇名出现的顺序
        for i in range(self.N):
            for j in range(self.K):
                if self.predict[i] == j:
                    self.predict[i] = ordered.index(j)  # 将预测结果有序化，增加预测准确率
                    break

        self.predict_list.append(self.predict)

    def calAcc(self):
        accuracy = np.sum(self.predict == self.data["label"]) / self.N
        self.accuracy_list.append(round(accuracy * 100, 2))

    def calLoss(self):
        loss = 0.5 * np.sum((self.predict - self.data["label"]) * (self.predict - self.data["label"]))
        self.loss_list.append(loss)

    def updateCenter(self):
        class_cnt = np.zeros([self.K, self.M])
        self.center_set = np.zeros([self.K, self.M])
        for i in range(self.N):
            class_cnt[self.predict[i]] += 1
            self.center_set[self.predict[i]] += self.data["features"][i]
        self.center_set /= class_cnt
        # print("中心集：")
        # print(self.center_set)

    def train(self):
        self.calDist()
        self.classify()
        self.calAcc()
        self.calLoss()
        last_predict = None
        if self.method == 1:
            print("使用欧氏距离时：")
        elif self.method == 2:
            print("使用马氏距离时：")
        elif self.method == 3:
            print("使用余弦距离时：")
        elif self.method == 4:
            print("使用L1距离时：")
        elif self.method == 5:
            print("使用L2距离时：")
        while ~(np.all(self.predict == last_predict)):
            last_predict = self.predict
            self.epochs += 1
            self.updateCenter()
            self.calDist()
            self.classify()
            self.calAcc()
            self.calLoss()
        # print("损失值列表",self.loss_list)
        # print("正确率列表",self.accuracy_list)
        print("损失值：", self.loss_list[-1])
        print("正确率：", self.accuracy_list[-1], "%")

    def visualize(self):
        fig, axes = plt.subplots(1, 3)
        fig.set_figheight(3)
        fig.set_figwidth(9)
        fig.suptitle("KMeans聚类对Yale数据集的分类")
        axes[0].set_title("分类结果变化图")
        axes[0].set_xlabel("人脸")
        axes[0].set_ylabel("表情")
        axes[0].set_xlim([-1, self.K])
        axes[0].set_ylim([-1, self.N / self.K])
        axes[0].grid()
        axes[0].xaxis.set_major_locator(MultipleLocator(1))
        axes[0].yaxis.set_major_locator(MultipleLocator(1))
        axes[1].set_title("损失值变化图")
        axes[1].set_xlabel("迭代数")
        axes[1].set_ylabel("损失值")
        axes[1].xaxis.set_major_locator(MultipleLocator(1))
        axes[2].set_title("正确率变化图")
        axes[2].set_xlabel("迭代数")
        axes[2].set_ylabel("正确率(%)")
        axes[2].xaxis.set_major_locator(MultipleLocator(1))
        epochs_list = []
        loss_list = []
        acc_list = []
        plt.tight_layout()
        plt.ion()
        for i in range(self.epochs):
            epochs_list.append(i + 1)
            loss_list.append(self.loss_list[i])
            acc_list.append(self.accuracy_list[i])
            try:
                axes[0].lines.remove(line_0[0])
                axes[1].lines.remove(line_1[0])
                axes[2].lines.remove(line_2[0])
            except Exception:
                pass
            x, y = [], []
            for j in range(self.N):
                x.append(j % self.K)
                y.append(int(j / self.K))
            line_0 = axes[0].scatter(x, y, c=self.predict_list[i])
            line_1 = axes[1].plot(epochs_list, loss_list, color=(0, 0, 0))
            line_2 = axes[2].plot(epochs_list, acc_list, color=(0, 0, 0))
            plt.savefig('新生成图片.jpg')
            plt.pause(1)


if __name__ == "__main__":
    Data = readData()
    model = K_Means(15, Data, 1)
