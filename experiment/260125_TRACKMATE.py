import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
from skimage import filters, measure
from skimage.segmentation import clear_border
import os

class TrackMate2DCellTracker:
    def __init__(self, max_displacement=10, min_cell_area=20, max_cell_area=500):
        self.max_displacement = max_displacement
        self.min_cell_area = min_cell_area
        self.max_cell_area = max_cell_area
        self.current_frame = 0
        self.tracks = {}
        self.next_track_id = 0
        self.frame_centroids = []

    def detect_cells(self, image):
        # 优化后的检测逻辑
        blurred = cv2.GaussianBlur(image, (7, 7), 1.5)
        binary = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=15,
            C=8
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        labels = measure.label(binary, connectivity=2)
        regions = measure.regionprops(labels)
        
        centroids = []
        for region in regions:
            if self.min_cell_area < region.area < self.max_cell_area:
                cx, cy = region.centroid
                centroids.append((float(cx), float(cy)))
        
        if len(centroids) == 0:
            print(f"警告：第 {self.current_frame} 帧未检测到任何细胞")
        return centroids

    def _calculate_cost_matrix(self, prev_centroids, curr_centroids):
        cost_matrix = np.zeros((len(prev_centroids), len(curr_centroids)))
        for i, (x1, y1) in enumerate(prev_centroids):
            for j, (x2, y2) in enumerate(curr_centroids):
                distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
                cost_matrix[i, j] = distance if distance < self.max_displacement else np.inf
        return cost_matrix

    def link_cells(self, prev_centroids, curr_centroids):
        self.frame_centroids.append(curr_centroids)
        
        if len(prev_centroids) == 0:
            for centroid in curr_centroids:
                self.tracks[self.next_track_id] = [(self.current_frame, centroid[0], centroid[1])]
                self.next_track_id += 1
            return
        
        cost_matrix = self._calculate_cost_matrix(prev_centroids, curr_centroids)
        
        # 新增容错逻辑
        if np.all(cost_matrix == np.inf):
            print(f"警告：第 {self.current_frame-1} 帧和第 {self.current_frame} 帧无可行匹配，为当前帧细胞新建轨迹")
            for centroid in curr_centroids:
                self.tracks[self.next_track_id] = [(self.current_frame, centroid[0], centroid[1])]
                self.next_track_id += 1
            return
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # 优化track_id查找逻辑
        prev_frame_track_map = {
            (track[-1][1], track[-1][2]): tid 
            for tid, track in self.tracks.items() 
            if track[-1][0] == self.current_frame - 1
        }
        
        matched_curr = set()
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < np.inf:
                prev_track_id = prev_frame_track_map.get((prev_centroids[r][0], prev_centroids[r][1]))
                if prev_track_id is not None:
                    self.tracks[prev_track_id].append((self.current_frame, curr_centroids[c][0], curr_centroids[c][1]))
                    matched_curr.add(c)
        
        for i, centroid in enumerate(curr_centroids):
            if i not in matched_curr:
                self.tracks[self.next_track_id] = [(self.current_frame, centroid[0], centroid[1])]
                self.next_track_id += 1

    # 保留process_sequence、draw_track_on_frame、export_tracking_video函数（不变）
    def process_sequence(self, image_sequence):
        """
        处理完整的2D时序图像序列
        :param image_sequence: 列表，每个元素是2D灰度图像（np.array）
        :return: 跟踪轨迹字典
        """
        # 重置跟踪状态
        self.current_frame = 0
        self.tracks = {}
        self.next_track_id = 0
        self.frame_centroids = []
        
        prev_centroids = []
        for frame_idx, image in enumerate(image_sequence):
            self.current_frame = frame_idx
            # 1. 检测当前帧细胞
            curr_centroids = self.detect_cells(image)
            # 2. 帧间链接
            self.link_cells(prev_centroids, curr_centroids)
            # 更新上一帧检测结果
            prev_centroids = curr_centroids
        
        return self.tracks

    def draw_track_on_frame(self, frame_img, frame_idx, track_history_frames=10):
        """
        在单帧图像上绘制跟踪轨迹
        :param frame_img: 原始2D图像
        :param frame_idx: 当前帧索引
        :param track_history_frames: 绘制轨迹的历史帧数
        :return: 绘制后的彩色图像
        """
        # 转换为彩色图像用于绘制
        if len(frame_img.shape) == 2:
            frame_color = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2BGR)
        else:
            frame_color = frame_img.copy()
        
        # 生成轨迹颜色（固定track_id对应固定颜色）
        np.random.seed(42)  # 固定随机种子保证颜色一致
        track_colors = {tid: (int(np.random.randint(0, 255)), 
                              int(np.random.randint(0, 255)), 
                              int(np.random.randint(0, 255))) 
                        for tid in self.tracks.keys()}
        
        # 绘制每个轨迹
        for track_id, points in self.tracks.items():
            # 筛选当前帧及历史帧的轨迹点
            track_points = [p for p in points if frame_idx - track_history_frames <= p[0] <= frame_idx]
            if len(track_points) < 1:
                continue
            
            # 绘制轨迹线
            for i in range(1, len(track_points)):
                prev_frame, x1, y1 = track_points[i-1]
                curr_frame, x2, y2 = track_points[i]
                # 只绘制当前帧相关的轨迹段
                if curr_frame == frame_idx:
                    cv2.line(frame_color, (int(y1), int(x1)), (int(y2), int(x2)), 
                             track_colors[track_id], 2)
            
            # 绘制当前帧细胞质心和track_id
            current_point = [p for p in points if p[0] == frame_idx]
            if current_point:
                x, y = current_point[0][1], current_point[0][2]
                # 绘制质心圆
                cv2.circle(frame_color, (int(y), int(x)), 3, track_colors[track_id], -1)
                # 绘制track_id文本
                cv2.putText(frame_color, f"ID:{track_id}", (int(y)+5, int(x)-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, track_colors[track_id], 1)
        
        return frame_color

    def export_tracking_video(self, image_sequence, output_path, fps=10, track_history_frames=10):
        """
        导出跟踪结果视频
        :param image_sequence: 原始图像序列
        :param output_path: 视频输出路径（如 "tracking_result.mp4"）
        :param fps: 视频帧率
        :param track_history_frames: 绘制轨迹的历史帧数
        """
        # 获取图像尺寸
        height, width = image_sequence[0].shape[:2]
        # 设置视频编码器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not video_writer.isOpened():
            raise ValueError(f"无法创建视频文件：{output_path}，请检查路径是否合法")
        
        # 逐帧绘制并写入视频
        print(f"开始导出视频，共 {len(image_sequence)} 帧...")
        for frame_idx, frame_img in enumerate(image_sequence):
            # 绘制跟踪结果
            frame_with_tracks = self.draw_track_on_frame(frame_img, frame_idx, track_history_frames)
            # 写入视频
            video_writer.write(frame_with_tracks)
            
            # 打印进度
            if (frame_idx + 1) % 20 == 0:
                print(f"已处理 {frame_idx + 1}/{len(image_sequence)} 帧")
        
        # 释放资源
        video_writer.release()
        print(f"视频已成功导出至：{os.path.abspath(output_path)}")

# -------------------------- 数据加载与运行代码 --------------------------
def load_image_stack(file_path=None, image_folder=None):
    """
    加载2D时序图像栈（支持两种方式）
    方式1：加载单文件多帧图像（如tif stack）
    方式2：加载文件夹内按顺序命名的图像序列（如 frame_001.tif, frame_002.tif）
    :param file_path: 单文件图像栈路径（优先）
    :param image_folder: 图像文件夹路径
    :return: 图像序列列表 [frame0, frame1, ...]
    """
    image_sequence = []
    
    # 方式1：加载单文件多帧图像（推荐，如ImageJ保存的tif stack）
    if file_path and os.path.exists(file_path):
        from skimage import io
        # 读取多帧tif
        img_stack = io.imread(file_path)
        # 确保是 (frames, height, width) 格式
        if len(img_stack.shape) == 3:
            for frame in img_stack:
                image_sequence.append(frame.astype(np.uint8))
        else:
            raise ValueError("图像栈格式错误，应为 (frames, height, width)")
    
    # 方式2：加载文件夹内的图像序列
    elif image_folder and os.path.exists(image_folder):
        # 获取文件夹内所有图像文件并按名称排序
        img_files = sorted([f for f in os.listdir(image_folder) 
                          if f.lower().endswith(('.tif', '.tiff', '.png', '.jpg', '.jpeg'))])
        if not img_files:
            raise ValueError("文件夹内未找到图像文件")
        
        for img_file in img_files:
            img_path = os.path.join(image_folder, img_file)
            frame = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if frame is not None:
                image_sequence.append(frame)
            else:
                print(f"警告：无法读取图像文件 {img_file}")
    
    else:
        raise ValueError("请提供有效的图像栈文件路径或图像文件夹路径")
    
    print(f"成功加载 {len(image_sequence)} 帧图像")
    return image_sequence



# 新增检测结果可视化函数
def visualize_detection_result(image_sequence, tracker, frame_indices=[0, 20, 50]):
    plt.figure(figsize=(15, 5))
    for i, frame_idx in enumerate(frame_indices):
        frame = image_sequence[frame_idx]
        # 临时设置current_frame，避免检测函数警告帧号错误
        tracker.current_frame = frame_idx
        centroids = tracker.detect_cells(frame)
        
        plt.subplot(1, len(frame_indices), i+1)
        plt.imshow(frame, cmap='gray')
        plt.scatter([y for x, y in centroids], [x for x, y in centroids], c='red', s=20)
        plt.title(f"Frame {frame_idx} (Detected: {len(centroids)})")
        plt.axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 配置参数（重点调整MAX_DISPLACEMENT）
    IMAGE_STACK_PATH = "D:/viterbi linkage/dataset/190621_++2_S02.tif"
    MAX_DISPLACEMENT = 100  # 关键调整：增大最大位移
    MIN_CELL_AREA = 20
    MAX_CELL_AREA = 800
    OUTPUT_VIDEO_PATH = "D:/viterbi linkage/dataset/cell_tracking_result.mp4"
    VIDEO_FPS = 120
    TRACK_HISTORY_FRAMES = 8
    
    try:
        image_sequence = load_image_stack(file_path=IMAGE_STACK_PATH)
        tracker = TrackMate2DCellTracker(
            max_displacement=MAX_DISPLACEMENT,
            min_cell_area=MIN_CELL_AREA,
            max_cell_area=MAX_CELL_AREA
        )
        
        # 新增：可视化检测结果
        visualize_detection_result(image_sequence, tracker, frame_indices=[0, 30, 60, 90])
        
        print("开始细胞跟踪...")
        tracks = tracker.process_sequence(image_sequence)
        print(f"跟踪完成！共检测到 {len(tracks)} 条细胞轨迹")
        
        tracker.export_tracking_video(
            image_sequence=image_sequence,
            output_path=OUTPUT_VIDEO_PATH,
            fps=VIDEO_FPS,
            track_history_frames=TRACK_HISTORY_FRAMES
        )
        
        print("\n轨迹统计信息：")
        for track_id, points in tracks.items():
            print(f"轨迹 {track_id}: 持续 {len(points)} 帧，起始位置 {tuple(map(int, points[0][1:]))}，结束位置 {tuple(map(int, points[-1][1:]))}")
    
    except Exception as e:
        print(f"运行出错：{str(e)}")
        # 新增：打印详细错误信息
        import traceback
        traceback.print_exc()