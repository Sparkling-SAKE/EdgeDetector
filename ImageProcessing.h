
#pragma once

#include <opencv2/imgcodecs.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <unordered_map>

#define PI	3.14159265358979

using namespace cv;
using namespace std;

/*****************
   함수 전방 선언
******************/
Mat ExtractLuma(Mat input);
double Gaussian(int x, int y, int sigma);
vector<vector<double>> GaussianBlurring(Mat input, int kernelSize, int sigma);
vector<vector<double>> Padding(vector<vector<double>> input, int paddingSize);
vector<vector<double>> Filter2D(vector<vector<double>> input, vector<vector<double>> mask);
vector<vector<double>> CalculateGradient(vector<vector<double>> input, vector<vector<double>>& angle, string filter);
Mat EdgeDetector(Mat input);
Mat Hysteresis(vector<vector<double>> input, int upperThreshold, int lowerThreshold);
Mat SaturationCasting(vector<vector<double>> input);
vector<vector<double>> NonMaximumSuppression(vector<vector<double>>& input, vector<vector<double>>& angle);


/***************************
* EdgeDetector(Mat input) -> Mat
* parameter
*	input : Edge를 찾을 RGB이미지
* return : Edge 이미지
***************************/
Mat EdgeDetector(Mat input)
{
	// Luma로 변환
	Mat lumaImage = std::move(ExtractLuma(std::move(input)));
	// 블러링
	vector<vector<double>> blurredImage = std::move(GaussianBlurring(std::move(lumaImage), 5, 3));
	// 각도 벡터 선언
	vector<vector<double>> angle(blurredImage.size(), vector<double>(blurredImage[0].size(), 0.0));
	// 그래디언트와 각도 계산
	vector<vector<double>> gradientImage = std::move(CalculateGradient(std::move(blurredImage), angle, "sobel"));
	// Nun-maximum Suppression 수행
	vector<vector<double>> afterNonMaximumSuppression = std::move(NonMaximumSuppression(gradientImage, angle));
	// double Thresholding과 Hysteresis를 통해 최종 에지 판단
	Mat result = std::move(Hysteresis(std::move(afterNonMaximumSuppression), 130, 100));

	// 결과 반환
	return result;
}


/***************************
* CalculateGradient(vector<vector<double>> input, vector<vector<double>>& angle, string filter) -> vector<vector<double>>
* parameter
*	input : gradient 계산을 위한 Blurring 된 이미지(벡터)
*	angle : gradient를 계산하는 과정에서 angle를 저장하기 위한 벡터
*	filter : soble / scharr 마스크 선택 가능
*   return : 이미지(벡터)의 Gradient 이미지(벡터)
***************************/
vector<vector<double>> CalculateGradient(vector<vector<double>> input, vector<vector<double>>& angle, string filter)
{
	// 3 * 3 mask 사용
	vector<vector<double>> maskX;
	vector<vector<double>> maskY;
	// x 축 방향 미분 필터 정의
	if (filter == "scharr")
		maskX = { {-3, 0, 3},
				  {-10, 0, 10},
				  {-3, 0, 3} };
	else if (filter == "sobel")
		maskX = { {-1, 0, 1},
				  {-2, 0, 2},
			      {-1, 0, 1} };
	// y 축 방향 미분 필터 정의
	if (filter == "scharr")
		maskY = { {-3, -10, -3},
				  {0, 0, 0},
				  {3, 10, 3} };
	else if (filter == "sobel")
		maskY = { {-1, -2, -1},
				  {0, 0, 0},
				  {1, 2, 1} };

	// xGradient 값 계산(음수가 포함되어 있을 수 있으므로 벡터 값으로 반환)
	vector<vector<double>> xGradient = Filter2D(input, maskX);
	// yGradient 값 계산(음수가 포함되어 있을 수 있으므로 벡터 값으로 반환)
	vector<vector<double>> yGradient = Filter2D(input, maskY);

	// xGradient와 yGradient이미지를 통해 구할 magnitude 영상 초기화
	int rows = input.size();	// row 크기
	int cols = input[0].size(); // col 크기
	vector<vector<double>> gradientMagnitude(rows, vector<double>(cols, 0));	// 결과 벡터 초기화

	// 모든 픽셀에 대해 Gradient값을 구하기 위한 루프
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			// angle 벡터에 현재 픽셀의 gradient 각도 저장, 라디안 값을 반환하므로 angle로 변환하기 위해 180/PI를 곱해줌.
			angle[i][j] = std::atan2(yGradient[i][j], xGradient[i][j]) * (180.0 / PI);
			gradientMagnitude[i][j] = abs(xGradient[i][j]) + abs(yGradient[i][j]); // gradient값의 magnitude를 L1 Norm 이용해서 저장
		}
	}

	// gradient 영상(벡터) 반환 
	return gradientMagnitude;
}

/***************************
* Hyteresis(vector<vector<double>> input, int upperThreshold, int lowerThreshold) -> Mat
* parameter
*	input : Non-maximum suppression된 영상(벡터)
*   upperThreshold : Strong을 구별할 Threshold
*   lowerThreshold : Weak를 구별할 Threshold
* return : 최종 Edge 이미지
***************************/
Mat Hysteresis(vector<vector<double>> input, int upperThreshold, int lowerThreshold)
{
	// threshold에 따라 상태를 구분하기 위해 enum class 선언
	enum class HYSTERESIS_TYPE
	{
		HT_NONE = 0,
		HT_WEAK,
		HT_UNKNOWN,
		HT_STRONG,
		HT_END
	};

	int rows = input.size();
	int cols = input[0].size();

	// hyteresis를 수행한 결과 이미지 선언
	Mat result = Mat_<uchar>(rows, cols);
	// double thresholding을 적용할 결과 벡터 선언
	vector<vector<pair<int, HYSTERESIS_TYPE>>> thresholding(rows, vector<pair<int, HYSTERESIS_TYPE>>(cols, make_pair(0, HYSTERESIS_TYPE::HT_NONE)));
	// 입력 영상(Non-maximum Suppression을 수행한 영상) 픽셀 값에 대해 double thresholding을 수행
	// upperThreshold보다 크면 <255, HT_STRONG>, lowerThreshold보다 작으면 <0, HT_WEAK>, 중간에 있으면 <픽셀 값, HT_UNKNOWN>으로 저장
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			int pixelValue = input[i][j];
			if (pixelValue >= upperThreshold)
				thresholding[i][j] = make_pair(255, HYSTERESIS_TYPE::HT_STRONG);
			else if (pixelValue < lowerThreshold)
				thresholding[i][j] = make_pair(0, HYSTERESIS_TYPE::HT_WEAK);
			else
				thresholding[i][j] = make_pair(pixelValue, HYSTERESIS_TYPE::HT_UNKNOWN);
		}
	}

	// HT_UNKNOWN인 픽셀에 대해서 주변(여덟 방향)에 HT_STRONG인 픽셀이 있으면 HT_STRONG으로 바꿔주고
	// 없으면 HT_WEAK 픽셀로 간주
	// 여덟 방향 벡터 초기화, 12시 방향부터 시계방향으로 선언
	vector<int> xDirection = { -1, -1, 0, 1, 1, 1, 0, -1 };
	vector<int> yDirection = { 0, 1, 1, 1, 0, -1, -1, -1 };
	// 경계값 처리를 안하기 위해 i, j를 1부터 rows - 1, cols - 1까지 반복
	for (int l = 0; l < 10; ++l)
	{
		for (int i = 1; i < rows - 1; ++i)
		{
			for (int j = 1; j < cols - 1; ++j)
			{
				if (thresholding[i][j].second == HYSTERESIS_TYPE::HT_STRONG)
					result.data[i * cols + j] = saturate_cast<uchar>(INT_MAX);
				else if (thresholding[i][j].second == HYSTERESIS_TYPE::HT_UNKNOWN)
				{
					for (int k = 0; k < 8; ++k)
					{
						if (thresholding[i + xDirection[k]][j + yDirection[k]].second == HYSTERESIS_TYPE::HT_STRONG)
						{
							thresholding[i][j].second = HYSTERESIS_TYPE::HT_STRONG;
							result.data[i * cols + j] = saturate_cast<uchar>(INT_MAX);
							break;
						}
					}
				}
				else
					result.data[i * cols + j] = saturate_cast<uchar>(0);
			}
		}
	}
	// 결과 이미지 반환
	return result;
}

/***************************
* ExtractLuma(Mat input) -> Mat
* parameter
*	input : 변환을 위한 RGB 영상
* return : RGB이미지에서 Luminance만 뽑은 Y 영상
***************************/
Mat ExtractLuma(Mat input)
{
	Mat luma = Mat_<uchar>(input.rows, input.cols);		// Y 이미지
	uchar* inputData = input.data;						// input 이미지의 data(픽셀 값)에 접근하기 위한 변수
	uchar* lumaData = luma.data;						// Y 이미지(luma 변수)의 data(픽셀 값)에 접근하기 위한 변수

	// opencv는 이미지를 읽으면 BGR 순으로 밝기값이 저장되므로 그에 맞게 순서 지정
	const int B = 0;
	const int G = 1;
	const int R = 2;

	int inputPixelLocation = 0;	// input 이미지의 픽셀 인덱스
	int lumaPixelLocation = 0;	// luma 이미지의 픽셀 인덱스

	// input이미지 픽셀 전체에 대해서
	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			/*
				BGRBGRBGRBGR... 순서로 밝기값 저장됨.
				input.cols * 3 => 한 행 => (input.cols * 3) * i 하면 i열로 이동
				+ (j * 3) => 해당 열에서 j 행 이동
				+ (R or G or B) => 해당 pixel의 원하는 칼라 밝기값
			*/
			inputPixelLocation = (i * input.cols * 3) + (j * 3); // input이미지의 [i][j]로 이동. BGR 색공간 이므로 3칸씩 이동해야함
			lumaPixelLocation = i * input.cols + j; // luma이미지의 [i][j]로 이동 Y 색공간 이므로 1칸씩 이동해야 함

			// Y = 0.299R + 0.587G + 0.114B
			// 위 식을 통해 계산된 Y값을 saturate_cast를 통해 uchar범위에 맞게 조정
			lumaData[lumaPixelLocation] = saturate_cast<uchar>(0.114 * inputData[inputPixelLocation + B] /* Blue */
				+ 0.587 * inputData[inputPixelLocation + G] /* Green */
				+ 0.299 * inputData[inputPixelLocation + R]); /* Red */
		}
	}
	// RGB -> Y 변환된 이미지 반환
	return luma;
}


/***************************
* Gaussian(int x, int sigma) -> double
* parameter
*	x : x좌표
*	y : y좌표
*	sigma : 가우시안의 표준편차
* return : 가우시안 분포 값
***************************/
double Gaussian(int x, int y, int sigma)
{
	// X~N(0, sigma^2)인 2D 가우시안 분포의 <x, y> 좌표에 대한 확률 값을 반환
	return exp(((x * x) + (y * y)) / (-2.0 * sigma * sigma));
}

/***************************
* GaussianBlurring(Mat input, int kernelSize, int sigma) -> vector<vector<double>>
* parameter
*	input : 입력 영상
*	kernelSize : 가우시안 커널 사이즈 (정사각형 커널 사용), 항상 홀수 가정
*	sigma : 가우시안의 표준편차 (x축과 y축에 동일한 표준편차 사용)
* return : 블러링된 영상
***************************/
vector<vector<double>> GaussianBlurring(Mat input, int kernelSize, int sigma)
{
	// 벡터에 Mat 타입 이미지 픽셀 값을 옮기는 과정
	vector<vector<double>> inputVec(input.rows, vector<double>(input.cols, 0.0));		// 밝기 값 벡터 초기화
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j)
			inputVec[i][j] = input.data[i * input.cols + j];		// 밝기 값 복사

	vector<vector<double>> mask(kernelSize, vector<double>(kernelSize, 0));		// 블러링을 위한 마스크 초기화
	int middle = kernelSize / 2;	// (0, 0)을 마스크의 중앙으로 잡기 위함
	// 마스크 전체를 돌면서 값을 저장하기 위함
	double maskSum = 0.0;	// 마스크 전체의 합을 1로 만들어 주기 위한 K 값
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			int xDistance = abs(i - middle);	// 가운데 부터 현재 위치의 x좌표가 얼마나 떨어져 있는지 계산
			int yDistance = abs(j - middle);	// 가운데 부터 현재 위치의 y좌표가 얼마나 떨어져 있는지 계산
			mask[i][j] = Gaussian(xDistance, yDistance, sigma);				// 가우시안 값을 마스크에 저장
			maskSum += Gaussian(xDistance, yDistance, sigma);				// 전체 합 누적
		}
	}

	// 저장된 마스크에 대해서 전체 마스크 합을 1로 만들어 주기 위해 각 마스크 값을 전체 합(maskSum)으로 나눠줌
	for (int i = 0; i < kernelSize; ++i)
		for (int j = 0; j < kernelSize; ++j)
			mask[i][j] /= maskSum;

	// Filter2D 함수를 통해 Blurring 수행
	vector<vector<double>> gaussianBlurredImage = std::move(Filter2D(inputVec, mask));

	// 블러링된 영상 반환
	return gaussianBlurredImage;
}

/***************************
* Padding(vector<vector<double>> input, int paddingSize) -> vector<vector<double>>
* parameter
*	input : 입력 영상
*	paddingSize : padding 사이즈
* return : 패딩된 영상
***************************/
vector<vector<double>> Padding(vector<vector<double>> input, int paddingSize)
{
	// 원래 이미지의 가로 / 세로 사이즈
	int rows = input.size();
	int cols = input[0].size();
	// 원래 이미지 사이즈에 paddingSize만큼이 상하좌우로 추가되므로 paddingSize * 2 만큼 더해준 paddedImage 생성
	vector<vector<double>> paddedImage(rows + paddingSize * 2, vector<double>(cols + paddingSize * 2, 0));
	// 패딩된 이미지의 가로 / 세로 사이즈
	int paddedRows = paddedImage.size();
	int paddedCols = paddedImage[0].size();

	// 패딩된 영역이 아닌 원래 이미지 영역에 원래 이미지 픽셀 값 저장
	// 모든 input 이미지 픽셀에 대해 수행, Y 채널이므로 한 칸 씩 이동하면서 저장
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			paddedImage[i + paddingSize][j + paddingSize] = input[i][j];

	// 이제 빈 공간에 패딩을 진행해야함
	// opencv의 BORDER_REFLECT_101 타입으로 진행하며, ...edcb|abcdefgh|gfed... 다음 방식과 같이 패딩이 진행됨.
	int symmetryAxisX = 0;				// x값 대칭 축
	int symmetryAxisY = 0;				// y값 대칭 축

	// 1. 상하좌우 패딩
	// i) 상단 : x좌표 - [0,paddingSize), y좌표 - [paddingSize, paddedImage.cols - paddingSize)
	//    대칭축 : paddedImageData의 (paddingSize)행
	// 상단 빈 공간에 대한 loop
	symmetryAxisY = paddingSize; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	for (int x = 0; x < paddingSize; ++x)
	{
		for (int y = paddingSize; y < paddedCols - paddingSize; ++y)
		{
			// 대칭축이 y축과 평행하므로 yDifference = 0
			int xDifference = abs(symmetryAxisY - x); // 축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[symmetryAxisY + xDifference][y];		// 픽셀 밝기값 복사
		}
	}
	// ii) 하단 : x좌표 - [paddedImage.rows - paddingSize, paddedImage.rows), y좌표 - [paddingSize, paddedImage.cols - paddingSize)
	//    대칭축 : paddedImageData의 (paddedImage.rows - paddingSize - 1)행
	// 하단 빈 공간에 대한 loop
	symmetryAxisY = paddedRows - paddingSize - 1; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	for (int x = paddedRows - paddingSize; x < paddedRows; ++x)
	{
		for (int y = paddingSize; y < paddedCols - paddingSize; ++y)
		{
			// 대칭축이 y축과 평행하므로 yDifference = 0
			int xDifference = abs(symmetryAxisY - x); // 축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[symmetryAxisY - xDifference][y];	// 픽셀 밝기값 복사
		}
	}
	// iii) 좌측 : x좌표 - [paddingSize, paddedImage.rows - paddingSize), y좌표 - [0, paddingSize)
	//    대칭축 : paddedImageData의 paddingSize열
	// 좌측 빈 공간에 대한 loop
	symmetryAxisX = paddingSize; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	for (int x = paddingSize; x < paddedRows - paddingSize; ++x)
	{
		for (int y = 0; y < paddingSize; ++y)
		{
			// 대칭축이 x축과 평행하므로 xDifference = 0
			int yDifference = abs(symmetryAxisX - y); // 축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[x][symmetryAxisX + yDifference]; // 픽셀 밝기값 복사
		}
	}
	// iv) 우측 : x좌표 - [paddingSize, paddedImage.rows - paddingSize), y좌표 - [paddedImage.cols - paddingSize, paddedImage.cols)
	//    대칭축 : paddedImageData의 (paddedImage.cols - paddingSize - 1)열
	// 우측 빈 공간에 대한 loop
	symmetryAxisX = paddedCols - paddingSize - 1; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	for (int x = paddingSize; x < paddedRows - paddingSize; ++x)
	{
		for (int y = paddedCols - paddingSize; y < paddedCols; ++y)
		{
			// 대칭축이 x축과 평행하므로 xDifference = 0
			int yDifference = abs(symmetryAxisX - y); // 축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[x][symmetryAxisX - yDifference]; // 픽셀 밝기값 복사
		}
	}

	// 2. 모서리 부분 패딩
	// i) 좌상 : x좌표 - [0,paddingSize), y좌표 - [0, paddingSize)
	//    대칭점 : paddedImageData의 <paddingSize, paddingSize>
	// 좌상단 빈 공간에 대한 loop
	symmetryAxisX = paddingSize; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	symmetryAxisY = paddingSize;
	for (int x = 0; x < paddingSize; ++x)
	{
		for (int y = 0; y < paddingSize; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y대칭축과 해당 픽셀과의 거리
			int yDifference = abs(symmetryAxisX - y); // X대칭축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[symmetryAxisY + xDifference][symmetryAxisX + yDifference];		// 픽셀 밝기값 복사
		}
	}
	// ii) 우상 : x좌표 - [0,paddingSize), y좌표 - [paddedImage.cols - paddingSize, paddedImage.cols)
	//    대칭점 : paddedImageData의 <paddingSize, paddedImage.cols - paddingSize - 1>
	// 우상단 빈 공간에 대한 loop
	symmetryAxisX = paddedCols - paddingSize - 1; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	symmetryAxisY = paddingSize;
	for (int x = 0; x < paddingSize; ++x)
	{
		for (int y = paddedCols - paddingSize; y < paddedCols; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y대칭축과 해당 픽셀과의 거리
			int yDifference = abs(symmetryAxisX - y); // X대칭축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[symmetryAxisY + xDifference][symmetryAxisX - yDifference];		// 픽셀 밝기값 복사
		}
	}
	// iii) 좌하 : x좌표 - [paddedImage.rows - paddingSize, paddedImage.rows), y좌표 - [0, paddingSize)
	//    대칭점 : paddedImageData의 <paddedImage.rows - paddingSize - 1, paddingSize>
	// 좌하단 빈 공간에 대한 loop
	symmetryAxisX = paddingSize; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	symmetryAxisY = paddedRows - paddingSize - 1;
	for (int x = paddedRows - paddingSize; x < paddedRows; ++x)
	{
		for (int y = 0; y < paddingSize; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y대칭축과 해당 픽셀과의 거리
			int yDifference = abs(symmetryAxisX - y); // X대칭축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[symmetryAxisY - xDifference][symmetryAxisX + yDifference];		// 픽셀 밝기값 복사
		}
	}
	// iv) 우하 : x좌표 - [paddedImage.rows - paddingSize, paddedImage.rows), y좌표 - [paddedImage.cols - paddingSize, paddedImage.cols)
	//    대칭점 : paddedImageData의 <paddedImage.rows - paddingSize - 1, paddedImage.cols - paddingSize - 1>
	// 우하단 빈 공간에 대한 loop
	symmetryAxisX = paddedCols - paddingSize - 1; // 위 주석에 의거한 대칭축 설정 및 루프 범위 설정
	symmetryAxisY = paddedRows - paddingSize - 1;
	for (int x = paddedRows - paddingSize; x < paddedRows; ++x)
	{
		for (int y = paddedCols - paddingSize; y < paddedCols; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y대칭축과 해당 픽셀과의 거리
			int yDifference = abs(symmetryAxisX - y); // X대칭축과 해당 픽셀과의 거리
			paddedImage[x][y] = paddedImage[symmetryAxisY - xDifference][symmetryAxisX - yDifference];		// 픽셀 밝기값 복사
		}
	}

	// 패딩된 영상 반환
	return paddedImage;
}


/***************************
* Filter2D(vector<vector<double>> input, vector<vector<double>> mask) -> vector<vector<double>>
* parameter
*	input : 입력 영상
*	mask : 컨볼루션 mask
* return : 입력 영상에 필터링을 수행한 결과 벡터
***************************/
vector<vector<double>> Filter2D(vector<vector<double>> input, vector<vector<double>> mask)
{
	// 입력 이미지의 가로 / 세로 사이즈
	int rows = input.size();
	int cols = input[0].size();
	vector<vector<double>> filteredVector(rows, vector<double>(cols, 0)); //  필터링 결과 벡터 초기화

	// 필터링을 위해 우선 패딩을 해주어야 함
	// paddingSize는 (mask(항상 정사각 행렬) 가로 길이(항상 홀수) - 1) / 2
	int paddingSize = (mask.size() - 1) / 2;
	vector<vector<double>> paddedImage = std::move(Padding(std::move(input), paddingSize));		// 패딩 수행
	// 패딩된 이미지의 가로 / 세로 사이즈
	int paddedRows = paddedImage.size();
	int paddedCols = paddedImage[0].size();

	// 필터링(컨볼루션) 수행
	// paddedImage의 원본 이미지 시작 지점인 (paddingSize, paddingSize) 부터 
	// (paddedImage.rows - paddingSize, paddedImage.cols - paddingSize)까지 반복문을 돌면 됨
	int middle = paddingSize; // (0, 0)을 중앙으로 잡기 위함
	for (int x = paddingSize; x < paddedRows - paddingSize; ++x)
	{
		for (int y = paddingSize; y < paddedCols - paddingSize; ++y)
		{
			double maskingPixelValue = 0.0;	// 컨볼루션 된 픽셀 값
			// 컨볼루션을 위한 반복문
			// 범위는 mask 사이즈가 항상 홀수로 가정했기 때문에 [-middle, middle]
			for (int i = (-1 * middle); i <= middle; ++i)
			{
				for (int j = (-1 * middle); j <= middle; ++j)
				{
					// 현재 픽셀을 (0, 0)으로 해서 (-middle, -middle)부터 순서대로 탐색
					// mask 중앙을 (0, 0)으로 해서 (middle, middle)부터 거꾸로 탐색하면서 가중합을 구함
					maskingPixelValue += (paddedImage[x + i][y + j] * mask[middle + (-1 * i)][middle + (-1 * j)]);
				}
			}
			// 계산된(필터링된) 픽셀 값을 저장
			// 필터링 된 이미지는 패딩된 이미지보다 paddingSize만큼 차이나기 때문에 빼주면서 저장해야함
			filteredVector[x - paddingSize][y - paddingSize] = maskingPixelValue;
		}
	}

	// 필터링된 영상 반환
	return filteredVector;
}

///***************************
//* NonMaximumSuppression(vector<vector<double>>& input, vector<vector<double>>& angle) -> vector<vector<double>
//* parameter
//*	input : 입력 영상(벡터)
//*	angle : gradient의 각도 벡터
//* return : Non-Maximum Suppression을 수행한 gradient 영상(벡터)
//***************************/
vector<vector<double>> NonMaximumSuppression(vector<vector<double>>& input, vector<vector<double>>& angle)
{
	// 입력 영상의 가로 / 세로 사이즈
	int rows = input.size();
	int cols = input[0].size();
	vector<vector<double>> result(input.size(), vector<double>(input[0].size(), 0));

	// 각도를 4 구역으로 나눠서 범위를 설정함
	const double FIRST_RANGE1 = 0.0;
	const double SECOND_RANGE = 22.5;
	const double SECOND_RANGE_NEG = -22.5;
	const double THIRD_RANGE = 67.5;
	const double THIRD_RANGE_NEG = -67.5;
	const double FOURTH_RANGE = 112.5;
	const double FOURTH_RANGE_NEG = -112.5;
	const double FIRST_RANGE2 = 157.5;
	const double FIRST_RANGE2_NEG = -157.5;
	const double ANGLE_LAST = 180.0;
	const double ANGLE_LAST_NEG = -180.0;

	// 모든 픽셀에 대해 반복
	// 항상 result 이미지와 input이미지의 현재 픽셀 좌표는 같으므로 result이미지에 저장할 때 inputPixelLocation 값 사용해도 됨
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			// 현재 픽셀 gradient 각도가 [-22.5, 0], [-180, -157.5), [0, 22.5), [157.5, 180]이면 ↔ 방향 비교
			if (FIRST_RANGE1 <= angle[i][j] < SECOND_RANGE || FIRST_RANGE2 <= angle[i][j] <= ANGLE_LAST
				|| SECOND_RANGE_NEG <= angle[i][j] <= FIRST_RANGE1 || ANGLE_LAST_NEG <= angle[i][j] < FIRST_RANGE2_NEG)
			{
				// 픽셀 범위 확인 -> ↔방향은 j값만 확인하면 됨
				if ((j - 1) < 0)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((j + 1) >= cols)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 주변 픽셀 모두보다 크면 살리고 아니면 0으로 죽임
					if (input[i][j] < input[i][j + 1] || input[i][j] < input[i][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
			// 현재 픽셀 gradient 각도가 [-157.5, -112.5), [22.5, 67.5)이면 ↙↗ 방향 비교
			else if (SECOND_RANGE <= angle[i][j] < THIRD_RANGE || FIRST_RANGE2_NEG <= angle[i][j] < FOURTH_RANGE)
			{
				// 픽셀 범위 확인
				if ((i - 1) < 0 || (j + 1) >= cols)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i + 1][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((i + 1) >= rows || (j - 1) < 0)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i - 1][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 주변 픽셀 모두보다 크면 살리고 아니면 0으로 죽임
					if (input[i][j] < input[i + 1][j - 1] || input[i][j] < input[i - 1][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
			// 현재 픽셀 gradient 각도가 [-112.5, -67.5),  [67.5, 112.5)이면 ↕ 방향 비교
			else if (THIRD_RANGE <= angle[i][j] < FOURTH_RANGE || FOURTH_RANGE_NEG <= angle[i][j] < THIRD_RANGE_NEG)
			{
				// 픽셀 범위 확인 -> ↕방향은 i값 범위만 확인하면 됨
				if ((i - 1) < 0)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i + 1][j])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((i + 1) >= rows)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i - 1][j])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 주변 픽셀 모두보다 크면 살리고 아니면 0으로 죽임
					if (input[i][j] < input[i + 1][j] || input[i][j] < input[i - 1][j])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
			// 현재 픽셀 gradient 각도가 [-67.5, -22.5) [112.5, 157.5)이면 ↖↘ 방향 비교
			else if (FOURTH_RANGE <= angle[i][j] < FIRST_RANGE2 || THIRD_RANGE_NEG <= angle[i][j] < SECOND_RANGE_NEG)
			{
				// 픽셀 범위 확인
				if ((i - 1) < 0 || (j - 1) < 0)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i + 1][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((i + 1) >= rows || (j + 1) >= cols)
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 작으면 0으로 죽이고, 크면 그대로 살림
					if (input[i][j] < input[i - 1][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// 현재 픽셀과 주변 픽셀 비교해서 현재 픽셀의 magnitude값이 주변 픽셀 모두보다 크면 살리고 아니면 0으로 죽임
					if (input[i][j] < input[i + 1][j + 1] || input[i][j] < input[i - 1][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
		}
	}

	// 결과 영상(벡터) 반환
	return result;
}

/***************************
* SaturationCasting(vector<vector<double>> input -> Mat
* parameter
*	input : 입력 영상(벡터)
* return : double 타입으로 저장된 벡터를 Mat_<uchar> 타입으로 변환해서 opencv에서 이미지를 띄울 수 있게 범위를 맞춰줌
***************************/
Mat SaturationCasting(vector<vector<double>> input)
{
	Mat castedImage = Mat_<uchar>(input.size(), input[0].size());
	for (int i = 0; i < castedImage.rows; ++i)
		for (int j = 0; j < castedImage.cols; ++j)
			castedImage.data[i * castedImage.cols + j] = saturate_cast<uchar>(input[i][j]);
	return castedImage;
}