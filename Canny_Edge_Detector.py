from PIL import Image #image를 다루는 python 라이브러리
import numpy as np #numpy 라이브러리

# 이미지 불러오기
img = Image.open("C:/Users/KimDaeWook/Desktop/컴퓨터비젼/FinalExam/harry.jpg")

#이미지 회색조로 변환
imgGray = img.convert('L')

# img를 array로 변환
numpy_img = np.array(imgGray)

# 1단계 노이즈 제거 

# 5*5 가우시안 필터 선언
kernel = np.array([[1, 4, 6, 4, 1],
                   [4, 16, 24, 16, 4],
                   [6, 24, 36, 24, 6],
                   [4, 16, 24, 16, 4],
                   [1, 4, 6, 4, 1]
                   ]) *(1/256)

# 컨볼루션 연산 함수 정의
def conv(img, filters, stride = 1, padding = 0): # 파라미터 : 이미지와, 필터(가우시안), 스트라이드(몇 칸씩 움직일지), 패딩의 크기
    x, y = img.shape # 입력 이미지의 모양 변수 선언
    filter_x, filter_y = filters.shape # 필터 이미지의 모양 변수 선언
    
    # 출력 이미지의 크기 공식
    out_x = (x + 2 * padding - filter_x) // stride + 1
    out_y = (y + 2 * padding - filter_y) // stride + 1
    
    # np.pad 이용하여 패딩으로 이미지 크기 유지
    in_img = np.pad(img, [(padding, padding), (padding, padding)], 'constant') 
    out = np.zeros((out_x, out_y)) # 출력 이미지 변수 선언
    
    # 이중 반복문을 이용한 컨볼루션 연산
    for x in range(out_x):
        x_start = x * stride
        x_end = x_start + filter_x
        for y in range(out_y):
            y_start = y * stride
            y_end = y_start + filter_y
            
            out[x, y] = np.sum(in_img[x_start:x_end, y_start:y_end] * filters)
    return out

# 5*5 가우시안 필터와 컨볼루션 연산 실행
smoothed_img = conv(numpy_img, kernel, 1, 2)

# 2단계 소벨 필터로 경계 및 그레디언트 방향 검출

# 소펠 필터 함수 정의
def sobel_filters(img):
    Gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32) # Gradient x sobel 커널 선언
    Gy = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32) # Gradient y sobel 커널 선언
    
    Ix = conv(img, Gx, 1, 1) # x 도함수 값
    Iy = conv(img, Gy, 1, 1) # y 도함수 값
    
    # 기울기의 크기 연산
    G = np.hypot(Ix, Iy) 
    G = G / G.max() * 255
    
    # 기울기 세타 연산
    theta = np.arctan2(Iy, Ix)
    
    return (G, theta)

# 소펠 필터 함수 실행
gradient_matrix, theta_matrix = sobel_filters(smoothed_img)

# 3단계 최대 값이 아닌 픽셀 0으로 만들기 
# 비최대치 억제 : 그레디언트 방향에서 검출된 경계 중 가장 큰 나머지는 제거

def Nonmax_suppression(img, theta):
    
    # 이미지 x, y 변수 선언
    x, y = img.shape
    
    # 출력 이미지 선언
    out = np.zeros((x, y), dtype=np.int32)
    
    # 기울기 각도로 변환
    angle = theta * 180. / np.pi
    angle[angle < 0] += 180

    # Non-Maximum suppression 진행
    for i in range(1, x-1):
        for j in range(1, y-1):
          
            a = 255
            b = 255
                
            # 0도 방향
            if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                a = img[i, j+1]
                b = img[i, j-1]
                
            # 45도 방향
            elif (22.5 <= angle[i, j] < 67.5):
                a = img[i+1, j-1]
                b = img[i-1, j+1]
                
            # 90도 방향
            elif (67.5 <= angle[i, j] < 112.5):
                a = img[i+1, j]
                b = img[i-1, j]
                
            # 135도 방향
            elif (112.5 <= angle[i, j] < 157.5):
                a = img[i-1, j-1]
                b = img[i+1, j+1]

            if (img[i,j] >= a) and (img[i,j] >= b):
                out[i,j] = img[i,j]
                
            else:
                out[i,j] = 0

            
    
    return out

# Nonmax_suppression 함수 실행
nonMaxImg = Nonmax_suppression(gradient_matrix, theta_matrix)


# 4단계 Hyteresis Thresholding : 두 개의 경계 값을 지정해서 
# 경계 영역에 있는 픽셀들 중 큰 경계 값 밖의 픽셀과 연결성이 없는 픽셀 제거

def Hysteresis_thresh(img, l_ThresholdRatio=0.01, h_ThresholdRatio=0.15):
    
    # Thresholds 부분
    h_Threshold = img.max() * h_ThresholdRatio # high Threshold 선언
    l_Threshold = h_Threshold * l_ThresholdRatio # low Threshold 선언
    
    x1, y1 = img.shape # 이미지 변수 선언
    thresh = np.zeros((x1, y1), dtype=np.int32) # threshold 결과 이미지 선언
    
    #Thresholds 연산 진행
    weak = np.int32(25)
    strong = np.int32(255)
    
    strong_i, strong_j = np.where(img >= h_Threshold)
    zeros_i, zeros_j = np.where(img < l_Threshold)
    
    weak_i, weak_j = np.where((img <= h_Threshold) & (img >= l_Threshold))
    
    thresh[strong_i, strong_j] = strong
    thresh[weak_i, weak_j] = weak
    
    
    # Hysteresis 부분
    x2, y2 = thresh.shape #threshold 결과를 사용하기 위해 변수 선언
    
    # hystersis 과정 수행
    for i in range(1, x2-1):
        for j in range(1, y2-1):
            if (thresh[i, j] == weak):
  
                if ((thresh[i+1, j-1] == strong) or (thresh[i+1, j] == strong) or (thresh[i+1, j+1] == strong)
                    or (thresh[i, j-1] == strong) or (thresh[i, j+1] == strong)
                    or (thresh[i-1, j-1] == strong) or (thresh[i-1, j] == strong) or (thresh[i-1, j+1] == strong)):
                    thresh[i, j] = strong
                else:
                    thresh[i, j] = 0
                
    
    return thresh

# Hysteresis_thresh 함수 실행
img_final = Hysteresis_thresh(nonMaxImg)

# 결과 출력
result=Image.fromarray(img_final)
result.show()