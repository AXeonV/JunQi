import cv2
import os
import numpy as np
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

def calc_pos(row, col, board_h):
	# 棋盘格点坐标
	margin_left, margin_top = 45, 45  # 棋盘左上角的偏移量
	cell_w, cell_h = 32, 20           # 每个格子的宽度和高度
	dis_w, dis_h = 107, 48            # 每个格子的水平和垂直间距

	if row in range(0, 6):
		center = (margin_left + col * dis_w + cell_w // 2, margin_top + row * dis_h + cell_h // 2)
	else:
		center = (margin_left + col * dis_w + cell_w // 2, board_h - margin_top - (11 - row) * dis_h - cell_h // 2)
	return center

def print_state(pieces, pic_idx, timestamp, last_steps=[]):
	# 加载背景棋盘图片
	board_img = cv2.imread('resources/board.jpg')
	board_w, board_h = board_img.shape[1], board_img.shape[0]

	cir_radius = 20                   # 棋子半径

	# 用opencv画圆
	for row, col, color, label in pieces:
		center = calc_pos(row, col, board_h)
		cv2.circle(board_img, center, cir_radius, color, -1)

	# print(last_steps)
	for step in last_steps:
		cv2.arrowedLine(board_img, calc_pos(step['from'][0], step['from'][1], board_h), calc_pos(step['to'][0], step['to'][1], board_h), (100, 255, 100), 5, tipLength=0.2)

	# 用PIL画汉字
	board_img_pil = Image.fromarray(cv2.cvtColor(board_img, cv2.COLOR_BGR2RGB))
	draw = ImageDraw.Draw(board_img_pil)
	font = ImageFont.truetype("resources/SIMLI.TTF", 36)

	for row, col, color, label in pieces:
		fill = (255, 255, 255)
		if last_steps != [] and (row, col) in [last_steps[0]['to'], last_steps[1]['to']]:
			fill = (255, 0, 0)
		center = calc_pos(row, col, board_h)
		bbox = draw.textbbox((0, 0), label, font=font)
		w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
		draw.text((center[0] - w // 2, center[1] - h * 5 // 6), label, font=font, fill=fill)

	# 转回OpenCV显示
	board_img = cv2.cvtColor(np.array(board_img_pil), cv2.COLOR_RGB2BGR)
	
	# 保存图片到 logs/时间戳/board.png
	save_dir = os.path.join('logs/', timestamp)
	os.makedirs(save_dir, exist_ok=True)
	save_path = os.path.join(save_dir, 'board' + str(pic_idx) + '.png')
	cv2.imwrite(save_path, board_img)
 
	# cv2.imshow('Board', board_img)
	# cv2.waitKey(0)
	# cv2.destroyAllWindows()
	
	
	
if __name__ == '__main__':
	pieces = [
		(0, 4, (255, 0, 0), '帅'),    # 红色棋子（汉字）
		(11, 0, (0, 0, 255), '卒'),   # 蓝色棋子（汉字）

	]
	print_state(pieces, 0, datetime.now().strftime('%Y%m%d.%H%M%S'))