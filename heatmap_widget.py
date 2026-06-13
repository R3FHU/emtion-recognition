import sys
import numpy as np
from PyQt6.QtWidgets import QWidget, QApplication, QVBoxLayout
from PyQt6.QtGui import QPainter, QColor
from PyQt6.QtCore import Qt, QRectF


class EmotionHeatmapWidget(QWidget):
    def __init__(self, history_length=50, parent=None):
        super().__init__(parent)
        # 7种情绪类别
        self.emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
        self.num_emotions = len(self.emotions)

        # 记录历史概率，矩阵形状为 [7, history_length]
        self.history_length = history_length
        self.prob_history = np.zeros((self.num_emotions, self.history_length))

        # 设置组件的最小尺寸
        self.setMinimumSize(400, 200)

    def update_heatmap(self, new_probs):
        """
        接收新的情绪概率向量并更新热力图
        :param new_probs: 长度为 7 的 numpy 数组或列表，值在 0~1 之间
        """
        # 将历史数据向左平移一位
        self.prob_history[:, :-1] = self.prob_history[:, 1:]
        # 将最新的概率赋值给最右侧
        self.prob_history[:, -1] = new_probs

        # 触发 paintEvent 重新绘制界面
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        width = self.width()
        height = self.height()

        # 预留左侧空间画文字标签
        label_width = 60
        grid_width = width - label_width

        # 计算每个小色块的宽度和高度
        cell_width = grid_width / self.history_length
        cell_height = height / self.num_emotions

        # 绘制热力图色块
        for row in range(self.num_emotions):
            # 绘制左侧情绪文本标签
            painter.setPen(QColor(200, 200, 200))  # 浅灰色字体，适合深色 UI
            text_rect = QRectF(0, row * cell_height, label_width, cell_height)
            painter.drawText(text_rect, Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight,
                             self.emotions[row] + " ")

            # 去除画笔边框，只填充颜色
            painter.setPen(Qt.PenStyle.NoPen)

            for col in range(self.history_length):
                prob = self.prob_history[row, col]

                # 根据概率计算颜色 (这里使用深蓝色到青蓝色的渐变，契合科技感 UI)
                # 概率越高，颜色越亮、透明度越高
                alpha = int(prob * 255)
                # RGB: (0, 150, 255) 是明亮的科技蓝
                color = QColor(0, 150, 255, alpha)

                painter.setBrush(color)

                # 计算方块的位置并绘制
                rect = QRectF(label_width + col * cell_width, row * cell_height, cell_width, cell_height)
                painter.drawRect(rect)


# ==========================================
# 下面是独立的测试代码，你可以直接运行它看效果
# ==========================================
if __name__ == '__main__':
    from PyQt6.QtCore import QTimer

    app = QApplication(sys.argv)

    # 模拟一个深色主题的面板
    main_window = QWidget()
    main_window.setStyleSheet("background-color: #1E1E1E;")
    layout = QVBoxLayout(main_window)

    # 实例化我们的热力图组件
    heatmap = EmotionHeatmapWidget(history_length=60)
    layout.addWidget(heatmap)


    # 模拟后台多线程源源不断发来的推流数据
    def mock_data_stream():
        # 随机生成7个概率并归一化，模拟真实模型的 Softmax 输出
        fake_probs = np.random.rand(7)
        fake_probs = np.exp(fake_probs) / np.sum(np.exp(fake_probs))

        # 让某一个情绪偶尔占据主导，看起来更逼真
        if np.random.rand() > 0.7:
            dominant = np.random.randint(0, 7)
            fake_probs[:] = 0.05
            fake_probs[dominant] = 0.7

        heatmap.update_heatmap(fake_probs)


    # 设置一个定时器，每 100 毫秒刷新一次数据（模拟 10 FPS 的推理速度）
    timer = QTimer()
    timer.timeout.connect(mock_data_stream)
    timer.start(100)

    main_window.resize(600, 300)
    main_window.show()
    sys.exit(app.exec())