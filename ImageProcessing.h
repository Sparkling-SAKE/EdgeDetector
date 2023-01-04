
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
   �Լ� ���� ����
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
*	input : Edge�� ã�� RGB�̹���
* return : Edge �̹���
***************************/
Mat EdgeDetector(Mat input)
{
	// Luma�� ��ȯ
	Mat lumaImage = std::move(ExtractLuma(std::move(input)));
	// ����
	vector<vector<double>> blurredImage = std::move(GaussianBlurring(std::move(lumaImage), 5, 3));
	// ���� ���� ����
	vector<vector<double>> angle(blurredImage.size(), vector<double>(blurredImage[0].size(), 0.0));
	// �׷����Ʈ�� ���� ���
	vector<vector<double>> gradientImage = std::move(CalculateGradient(std::move(blurredImage), angle, "sobel"));
	// Nun-maximum Suppression ����
	vector<vector<double>> afterNonMaximumSuppression = std::move(NonMaximumSuppression(gradientImage, angle));
	// double Thresholding�� Hysteresis�� ���� ���� ���� �Ǵ�
	Mat result = std::move(Hysteresis(std::move(afterNonMaximumSuppression), 130, 100));

	// ��� ��ȯ
	return result;
}


/***************************
* CalculateGradient(vector<vector<double>> input, vector<vector<double>>& angle, string filter) -> vector<vector<double>>
* parameter
*	input : gradient ����� ���� Blurring �� �̹���(����)
*	angle : gradient�� ����ϴ� �������� angle�� �����ϱ� ���� ����
*	filter : soble / scharr ����ũ ���� ����
*   return : �̹���(����)�� Gradient �̹���(����)
***************************/
vector<vector<double>> CalculateGradient(vector<vector<double>> input, vector<vector<double>>& angle, string filter)
{
	// 3 * 3 mask ���
	vector<vector<double>> maskX;
	vector<vector<double>> maskY;
	// x �� ���� �̺� ���� ����
	if (filter == "scharr")
		maskX = { {-3, 0, 3},
				  {-10, 0, 10},
				  {-3, 0, 3} };
	else if (filter == "sobel")
		maskX = { {-1, 0, 1},
				  {-2, 0, 2},
			      {-1, 0, 1} };
	// y �� ���� �̺� ���� ����
	if (filter == "scharr")
		maskY = { {-3, -10, -3},
				  {0, 0, 0},
				  {3, 10, 3} };
	else if (filter == "sobel")
		maskY = { {-1, -2, -1},
				  {0, 0, 0},
				  {1, 2, 1} };

	// xGradient �� ���(������ ���ԵǾ� ���� �� �����Ƿ� ���� ������ ��ȯ)
	vector<vector<double>> xGradient = Filter2D(input, maskX);
	// yGradient �� ���(������ ���ԵǾ� ���� �� �����Ƿ� ���� ������ ��ȯ)
	vector<vector<double>> yGradient = Filter2D(input, maskY);

	// xGradient�� yGradient�̹����� ���� ���� magnitude ���� �ʱ�ȭ
	int rows = input.size();	// row ũ��
	int cols = input[0].size(); // col ũ��
	vector<vector<double>> gradientMagnitude(rows, vector<double>(cols, 0));	// ��� ���� �ʱ�ȭ

	// ��� �ȼ��� ���� Gradient���� ���ϱ� ���� ����
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			// angle ���Ϳ� ���� �ȼ��� gradient ���� ����, ���� ���� ��ȯ�ϹǷ� angle�� ��ȯ�ϱ� ���� 180/PI�� ������.
			angle[i][j] = std::atan2(yGradient[i][j], xGradient[i][j]) * (180.0 / PI);
			gradientMagnitude[i][j] = abs(xGradient[i][j]) + abs(yGradient[i][j]); // gradient���� magnitude�� L1 Norm �̿��ؼ� ����
		}
	}

	// gradient ����(����) ��ȯ 
	return gradientMagnitude;
}

/***************************
* Hyteresis(vector<vector<double>> input, int upperThreshold, int lowerThreshold) -> Mat
* parameter
*	input : Non-maximum suppression�� ����(����)
*   upperThreshold : Strong�� ������ Threshold
*   lowerThreshold : Weak�� ������ Threshold
* return : ���� Edge �̹���
***************************/
Mat Hysteresis(vector<vector<double>> input, int upperThreshold, int lowerThreshold)
{
	// threshold�� ���� ���¸� �����ϱ� ���� enum class ����
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

	// hyteresis�� ������ ��� �̹��� ����
	Mat result = Mat_<uchar>(rows, cols);
	// double thresholding�� ������ ��� ���� ����
	vector<vector<pair<int, HYSTERESIS_TYPE>>> thresholding(rows, vector<pair<int, HYSTERESIS_TYPE>>(cols, make_pair(0, HYSTERESIS_TYPE::HT_NONE)));
	// �Է� ����(Non-maximum Suppression�� ������ ����) �ȼ� ���� ���� double thresholding�� ����
	// upperThreshold���� ũ�� <255, HT_STRONG>, lowerThreshold���� ������ <0, HT_WEAK>, �߰��� ������ <�ȼ� ��, HT_UNKNOWN>���� ����
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

	// HT_UNKNOWN�� �ȼ��� ���ؼ� �ֺ�(���� ����)�� HT_STRONG�� �ȼ��� ������ HT_STRONG���� �ٲ��ְ�
	// ������ HT_WEAK �ȼ��� ����
	// ���� ���� ���� �ʱ�ȭ, 12�� ������� �ð�������� ����
	vector<int> xDirection = { -1, -1, 0, 1, 1, 1, 0, -1 };
	vector<int> yDirection = { 0, 1, 1, 1, 0, -1, -1, -1 };
	// ��谪 ó���� ���ϱ� ���� i, j�� 1���� rows - 1, cols - 1���� �ݺ�
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
	// ��� �̹��� ��ȯ
	return result;
}

/***************************
* ExtractLuma(Mat input) -> Mat
* parameter
*	input : ��ȯ�� ���� RGB ����
* return : RGB�̹������� Luminance�� ���� Y ����
***************************/
Mat ExtractLuma(Mat input)
{
	Mat luma = Mat_<uchar>(input.rows, input.cols);		// Y �̹���
	uchar* inputData = input.data;						// input �̹����� data(�ȼ� ��)�� �����ϱ� ���� ����
	uchar* lumaData = luma.data;						// Y �̹���(luma ����)�� data(�ȼ� ��)�� �����ϱ� ���� ����

	// opencv�� �̹����� ������ BGR ������ ��Ⱚ�� ����ǹǷ� �׿� �°� ���� ����
	const int B = 0;
	const int G = 1;
	const int R = 2;

	int inputPixelLocation = 0;	// input �̹����� �ȼ� �ε���
	int lumaPixelLocation = 0;	// luma �̹����� �ȼ� �ε���

	// input�̹��� �ȼ� ��ü�� ���ؼ�
	for (int i = 0; i < input.rows; ++i)
	{
		for (int j = 0; j < input.cols; ++j)
		{
			/*
				BGRBGRBGRBGR... ������ ��Ⱚ �����.
				input.cols * 3 => �� �� => (input.cols * 3) * i �ϸ� i���� �̵�
				+ (j * 3) => �ش� ������ j �� �̵�
				+ (R or G or B) => �ش� pixel�� ���ϴ� Į�� ��Ⱚ
			*/
			inputPixelLocation = (i * input.cols * 3) + (j * 3); // input�̹����� [i][j]�� �̵�. BGR ������ �̹Ƿ� 3ĭ�� �̵��ؾ���
			lumaPixelLocation = i * input.cols + j; // luma�̹����� [i][j]�� �̵� Y ������ �̹Ƿ� 1ĭ�� �̵��ؾ� ��

			// Y = 0.299R + 0.587G + 0.114B
			// �� ���� ���� ���� Y���� saturate_cast�� ���� uchar������ �°� ����
			lumaData[lumaPixelLocation] = saturate_cast<uchar>(0.114 * inputData[inputPixelLocation + B] /* Blue */
				+ 0.587 * inputData[inputPixelLocation + G] /* Green */
				+ 0.299 * inputData[inputPixelLocation + R]); /* Red */
		}
	}
	// RGB -> Y ��ȯ�� �̹��� ��ȯ
	return luma;
}


/***************************
* Gaussian(int x, int sigma) -> double
* parameter
*	x : x��ǥ
*	y : y��ǥ
*	sigma : ����þ��� ǥ������
* return : ����þ� ���� ��
***************************/
double Gaussian(int x, int y, int sigma)
{
	// X~N(0, sigma^2)�� 2D ����þ� ������ <x, y> ��ǥ�� ���� Ȯ�� ���� ��ȯ
	return exp(((x * x) + (y * y)) / (-2.0 * sigma * sigma));
}

/***************************
* GaussianBlurring(Mat input, int kernelSize, int sigma) -> vector<vector<double>>
* parameter
*	input : �Է� ����
*	kernelSize : ����þ� Ŀ�� ������ (���簢�� Ŀ�� ���), �׻� Ȧ�� ����
*	sigma : ����þ��� ǥ������ (x��� y�࿡ ������ ǥ������ ���)
* return : ������ ����
***************************/
vector<vector<double>> GaussianBlurring(Mat input, int kernelSize, int sigma)
{
	// ���Ϳ� Mat Ÿ�� �̹��� �ȼ� ���� �ű�� ����
	vector<vector<double>> inputVec(input.rows, vector<double>(input.cols, 0.0));		// ��� �� ���� �ʱ�ȭ
	for (int i = 0; i < input.rows; ++i)
		for (int j = 0; j < input.cols; ++j)
			inputVec[i][j] = input.data[i * input.cols + j];		// ��� �� ����

	vector<vector<double>> mask(kernelSize, vector<double>(kernelSize, 0));		// ������ ���� ����ũ �ʱ�ȭ
	int middle = kernelSize / 2;	// (0, 0)�� ����ũ�� �߾����� ��� ����
	// ����ũ ��ü�� ���鼭 ���� �����ϱ� ����
	double maskSum = 0.0;	// ����ũ ��ü�� ���� 1�� ����� �ֱ� ���� K ��
	for (int i = 0; i < kernelSize; ++i)
	{
		for (int j = 0; j < kernelSize; ++j)
		{
			int xDistance = abs(i - middle);	// ��� ���� ���� ��ġ�� x��ǥ�� �󸶳� ������ �ִ��� ���
			int yDistance = abs(j - middle);	// ��� ���� ���� ��ġ�� y��ǥ�� �󸶳� ������ �ִ��� ���
			mask[i][j] = Gaussian(xDistance, yDistance, sigma);				// ����þ� ���� ����ũ�� ����
			maskSum += Gaussian(xDistance, yDistance, sigma);				// ��ü �� ����
		}
	}

	// ����� ����ũ�� ���ؼ� ��ü ����ũ ���� 1�� ����� �ֱ� ���� �� ����ũ ���� ��ü ��(maskSum)���� ������
	for (int i = 0; i < kernelSize; ++i)
		for (int j = 0; j < kernelSize; ++j)
			mask[i][j] /= maskSum;

	// Filter2D �Լ��� ���� Blurring ����
	vector<vector<double>> gaussianBlurredImage = std::move(Filter2D(inputVec, mask));

	// ������ ���� ��ȯ
	return gaussianBlurredImage;
}

/***************************
* Padding(vector<vector<double>> input, int paddingSize) -> vector<vector<double>>
* parameter
*	input : �Է� ����
*	paddingSize : padding ������
* return : �е��� ����
***************************/
vector<vector<double>> Padding(vector<vector<double>> input, int paddingSize)
{
	// ���� �̹����� ���� / ���� ������
	int rows = input.size();
	int cols = input[0].size();
	// ���� �̹��� ����� paddingSize��ŭ�� �����¿�� �߰��ǹǷ� paddingSize * 2 ��ŭ ������ paddedImage ����
	vector<vector<double>> paddedImage(rows + paddingSize * 2, vector<double>(cols + paddingSize * 2, 0));
	// �е��� �̹����� ���� / ���� ������
	int paddedRows = paddedImage.size();
	int paddedCols = paddedImage[0].size();

	// �е��� ������ �ƴ� ���� �̹��� ������ ���� �̹��� �ȼ� �� ����
	// ��� input �̹��� �ȼ��� ���� ����, Y ä���̹Ƿ� �� ĭ �� �̵��ϸ鼭 ����
	for (int i = 0; i < rows; ++i)
		for (int j = 0; j < cols; ++j)
			paddedImage[i + paddingSize][j + paddingSize] = input[i][j];

	// ���� �� ������ �е��� �����ؾ���
	// opencv�� BORDER_REFLECT_101 Ÿ������ �����ϸ�, ...edcb|abcdefgh|gfed... ���� ��İ� ���� �е��� �����.
	int symmetryAxisX = 0;				// x�� ��Ī ��
	int symmetryAxisY = 0;				// y�� ��Ī ��

	// 1. �����¿� �е�
	// i) ��� : x��ǥ - [0,paddingSize), y��ǥ - [paddingSize, paddedImage.cols - paddingSize)
	//    ��Ī�� : paddedImageData�� (paddingSize)��
	// ��� �� ������ ���� loop
	symmetryAxisY = paddingSize; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	for (int x = 0; x < paddingSize; ++x)
	{
		for (int y = paddingSize; y < paddedCols - paddingSize; ++y)
		{
			// ��Ī���� y��� �����ϹǷ� yDifference = 0
			int xDifference = abs(symmetryAxisY - x); // ��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[symmetryAxisY + xDifference][y];		// �ȼ� ��Ⱚ ����
		}
	}
	// ii) �ϴ� : x��ǥ - [paddedImage.rows - paddingSize, paddedImage.rows), y��ǥ - [paddingSize, paddedImage.cols - paddingSize)
	//    ��Ī�� : paddedImageData�� (paddedImage.rows - paddingSize - 1)��
	// �ϴ� �� ������ ���� loop
	symmetryAxisY = paddedRows - paddingSize - 1; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	for (int x = paddedRows - paddingSize; x < paddedRows; ++x)
	{
		for (int y = paddingSize; y < paddedCols - paddingSize; ++y)
		{
			// ��Ī���� y��� �����ϹǷ� yDifference = 0
			int xDifference = abs(symmetryAxisY - x); // ��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[symmetryAxisY - xDifference][y];	// �ȼ� ��Ⱚ ����
		}
	}
	// iii) ���� : x��ǥ - [paddingSize, paddedImage.rows - paddingSize), y��ǥ - [0, paddingSize)
	//    ��Ī�� : paddedImageData�� paddingSize��
	// ���� �� ������ ���� loop
	symmetryAxisX = paddingSize; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	for (int x = paddingSize; x < paddedRows - paddingSize; ++x)
	{
		for (int y = 0; y < paddingSize; ++y)
		{
			// ��Ī���� x��� �����ϹǷ� xDifference = 0
			int yDifference = abs(symmetryAxisX - y); // ��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[x][symmetryAxisX + yDifference]; // �ȼ� ��Ⱚ ����
		}
	}
	// iv) ���� : x��ǥ - [paddingSize, paddedImage.rows - paddingSize), y��ǥ - [paddedImage.cols - paddingSize, paddedImage.cols)
	//    ��Ī�� : paddedImageData�� (paddedImage.cols - paddingSize - 1)��
	// ���� �� ������ ���� loop
	symmetryAxisX = paddedCols - paddingSize - 1; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	for (int x = paddingSize; x < paddedRows - paddingSize; ++x)
	{
		for (int y = paddedCols - paddingSize; y < paddedCols; ++y)
		{
			// ��Ī���� x��� �����ϹǷ� xDifference = 0
			int yDifference = abs(symmetryAxisX - y); // ��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[x][symmetryAxisX - yDifference]; // �ȼ� ��Ⱚ ����
		}
	}

	// 2. �𼭸� �κ� �е�
	// i) �»� : x��ǥ - [0,paddingSize), y��ǥ - [0, paddingSize)
	//    ��Ī�� : paddedImageData�� <paddingSize, paddingSize>
	// �»�� �� ������ ���� loop
	symmetryAxisX = paddingSize; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	symmetryAxisY = paddingSize;
	for (int x = 0; x < paddingSize; ++x)
	{
		for (int y = 0; y < paddingSize; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y��Ī��� �ش� �ȼ����� �Ÿ�
			int yDifference = abs(symmetryAxisX - y); // X��Ī��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[symmetryAxisY + xDifference][symmetryAxisX + yDifference];		// �ȼ� ��Ⱚ ����
		}
	}
	// ii) ��� : x��ǥ - [0,paddingSize), y��ǥ - [paddedImage.cols - paddingSize, paddedImage.cols)
	//    ��Ī�� : paddedImageData�� <paddingSize, paddedImage.cols - paddingSize - 1>
	// ���� �� ������ ���� loop
	symmetryAxisX = paddedCols - paddingSize - 1; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	symmetryAxisY = paddingSize;
	for (int x = 0; x < paddingSize; ++x)
	{
		for (int y = paddedCols - paddingSize; y < paddedCols; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y��Ī��� �ش� �ȼ����� �Ÿ�
			int yDifference = abs(symmetryAxisX - y); // X��Ī��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[symmetryAxisY + xDifference][symmetryAxisX - yDifference];		// �ȼ� ��Ⱚ ����
		}
	}
	// iii) ���� : x��ǥ - [paddedImage.rows - paddingSize, paddedImage.rows), y��ǥ - [0, paddingSize)
	//    ��Ī�� : paddedImageData�� <paddedImage.rows - paddingSize - 1, paddingSize>
	// ���ϴ� �� ������ ���� loop
	symmetryAxisX = paddingSize; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	symmetryAxisY = paddedRows - paddingSize - 1;
	for (int x = paddedRows - paddingSize; x < paddedRows; ++x)
	{
		for (int y = 0; y < paddingSize; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y��Ī��� �ش� �ȼ����� �Ÿ�
			int yDifference = abs(symmetryAxisX - y); // X��Ī��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[symmetryAxisY - xDifference][symmetryAxisX + yDifference];		// �ȼ� ��Ⱚ ����
		}
	}
	// iv) ���� : x��ǥ - [paddedImage.rows - paddingSize, paddedImage.rows), y��ǥ - [paddedImage.cols - paddingSize, paddedImage.cols)
	//    ��Ī�� : paddedImageData�� <paddedImage.rows - paddingSize - 1, paddedImage.cols - paddingSize - 1>
	// ���ϴ� �� ������ ���� loop
	symmetryAxisX = paddedCols - paddingSize - 1; // �� �ּ��� �ǰ��� ��Ī�� ���� �� ���� ���� ����
	symmetryAxisY = paddedRows - paddingSize - 1;
	for (int x = paddedRows - paddingSize; x < paddedRows; ++x)
	{
		for (int y = paddedCols - paddingSize; y < paddedCols; ++y)
		{
			int xDifference = abs(symmetryAxisY - x); // Y��Ī��� �ش� �ȼ����� �Ÿ�
			int yDifference = abs(symmetryAxisX - y); // X��Ī��� �ش� �ȼ����� �Ÿ�
			paddedImage[x][y] = paddedImage[symmetryAxisY - xDifference][symmetryAxisX - yDifference];		// �ȼ� ��Ⱚ ����
		}
	}

	// �е��� ���� ��ȯ
	return paddedImage;
}


/***************************
* Filter2D(vector<vector<double>> input, vector<vector<double>> mask) -> vector<vector<double>>
* parameter
*	input : �Է� ����
*	mask : ������� mask
* return : �Է� ���� ���͸��� ������ ��� ����
***************************/
vector<vector<double>> Filter2D(vector<vector<double>> input, vector<vector<double>> mask)
{
	// �Է� �̹����� ���� / ���� ������
	int rows = input.size();
	int cols = input[0].size();
	vector<vector<double>> filteredVector(rows, vector<double>(cols, 0)); //  ���͸� ��� ���� �ʱ�ȭ

	// ���͸��� ���� �켱 �е��� ���־�� ��
	// paddingSize�� (mask(�׻� ���簢 ���) ���� ����(�׻� Ȧ��) - 1) / 2
	int paddingSize = (mask.size() - 1) / 2;
	vector<vector<double>> paddedImage = std::move(Padding(std::move(input), paddingSize));		// �е� ����
	// �е��� �̹����� ���� / ���� ������
	int paddedRows = paddedImage.size();
	int paddedCols = paddedImage[0].size();

	// ���͸�(�������) ����
	// paddedImage�� ���� �̹��� ���� ������ (paddingSize, paddingSize) ���� 
	// (paddedImage.rows - paddingSize, paddedImage.cols - paddingSize)���� �ݺ����� ���� ��
	int middle = paddingSize; // (0, 0)�� �߾����� ��� ����
	for (int x = paddingSize; x < paddedRows - paddingSize; ++x)
	{
		for (int y = paddingSize; y < paddedCols - paddingSize; ++y)
		{
			double maskingPixelValue = 0.0;	// ������� �� �ȼ� ��
			// ��������� ���� �ݺ���
			// ������ mask ����� �׻� Ȧ���� �����߱� ������ [-middle, middle]
			for (int i = (-1 * middle); i <= middle; ++i)
			{
				for (int j = (-1 * middle); j <= middle; ++j)
				{
					// ���� �ȼ��� (0, 0)���� �ؼ� (-middle, -middle)���� ������� Ž��
					// mask �߾��� (0, 0)���� �ؼ� (middle, middle)���� �Ųٷ� Ž���ϸ鼭 �������� ����
					maskingPixelValue += (paddedImage[x + i][y + j] * mask[middle + (-1 * i)][middle + (-1 * j)]);
				}
			}
			// ����(���͸���) �ȼ� ���� ����
			// ���͸� �� �̹����� �е��� �̹������� paddingSize��ŭ ���̳��� ������ ���ָ鼭 �����ؾ���
			filteredVector[x - paddingSize][y - paddingSize] = maskingPixelValue;
		}
	}

	// ���͸��� ���� ��ȯ
	return filteredVector;
}

///***************************
//* NonMaximumSuppression(vector<vector<double>>& input, vector<vector<double>>& angle) -> vector<vector<double>
//* parameter
//*	input : �Է� ����(����)
//*	angle : gradient�� ���� ����
//* return : Non-Maximum Suppression�� ������ gradient ����(����)
//***************************/
vector<vector<double>> NonMaximumSuppression(vector<vector<double>>& input, vector<vector<double>>& angle)
{
	// �Է� ������ ���� / ���� ������
	int rows = input.size();
	int cols = input[0].size();
	vector<vector<double>> result(input.size(), vector<double>(input[0].size(), 0));

	// ������ 4 �������� ������ ������ ������
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

	// ��� �ȼ��� ���� �ݺ�
	// �׻� result �̹����� input�̹����� ���� �ȼ� ��ǥ�� �����Ƿ� result�̹����� ������ �� inputPixelLocation �� ����ص� ��
	for (int i = 0; i < rows; ++i)
	{
		for (int j = 0; j < cols; ++j)
		{
			// ���� �ȼ� gradient ������ [-22.5, 0], [-180, -157.5), [0, 22.5), [157.5, 180]�̸� �� ���� ��
			if (FIRST_RANGE1 <= angle[i][j] < SECOND_RANGE || FIRST_RANGE2 <= angle[i][j] <= ANGLE_LAST
				|| SECOND_RANGE_NEG <= angle[i][j] <= FIRST_RANGE1 || ANGLE_LAST_NEG <= angle[i][j] < FIRST_RANGE2_NEG)
			{
				// �ȼ� ���� Ȯ�� -> ������� j���� Ȯ���ϸ� ��
				if ((j - 1) < 0)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((j + 1) >= cols)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� �ֺ� �ȼ� ��κ��� ũ�� �츮�� �ƴϸ� 0���� ����
					if (input[i][j] < input[i][j + 1] || input[i][j] < input[i][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
			// ���� �ȼ� gradient ������ [-157.5, -112.5), [22.5, 67.5)�̸� �ע� ���� ��
			else if (SECOND_RANGE <= angle[i][j] < THIRD_RANGE || FIRST_RANGE2_NEG <= angle[i][j] < FOURTH_RANGE)
			{
				// �ȼ� ���� Ȯ��
				if ((i - 1) < 0 || (j + 1) >= cols)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i + 1][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((i + 1) >= rows || (j - 1) < 0)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i - 1][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� �ֺ� �ȼ� ��κ��� ũ�� �츮�� �ƴϸ� 0���� ����
					if (input[i][j] < input[i + 1][j - 1] || input[i][j] < input[i - 1][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
			// ���� �ȼ� gradient ������ [-112.5, -67.5),  [67.5, 112.5)�̸� �� ���� ��
			else if (THIRD_RANGE <= angle[i][j] < FOURTH_RANGE || FOURTH_RANGE_NEG <= angle[i][j] < THIRD_RANGE_NEG)
			{
				// �ȼ� ���� Ȯ�� -> �չ����� i�� ������ Ȯ���ϸ� ��
				if ((i - 1) < 0)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i + 1][j])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((i + 1) >= rows)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i - 1][j])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� �ֺ� �ȼ� ��κ��� ũ�� �츮�� �ƴϸ� 0���� ����
					if (input[i][j] < input[i + 1][j] || input[i][j] < input[i - 1][j])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
			// ���� �ȼ� gradient ������ [-67.5, -22.5) [112.5, 157.5)�̸� �آ� ���� ��
			else if (FOURTH_RANGE <= angle[i][j] < FIRST_RANGE2 || THIRD_RANGE_NEG <= angle[i][j] < SECOND_RANGE_NEG)
			{
				// �ȼ� ���� Ȯ��
				if ((i - 1) < 0 || (j - 1) < 0)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i + 1][j + 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else if ((i + 1) >= rows || (j + 1) >= cols)
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� ������ 0���� ���̰�, ũ�� �״�� �츲
					if (input[i][j] < input[i - 1][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
				else
				{
					// ���� �ȼ��� �ֺ� �ȼ� ���ؼ� ���� �ȼ��� magnitude���� �ֺ� �ȼ� ��κ��� ũ�� �츮�� �ƴϸ� 0���� ����
					if (input[i][j] < input[i + 1][j + 1] || input[i][j] < input[i - 1][j - 1])
						result[i][j] = 0;
					else
						result[i][j] = input[i][j];
				}
			}
		}
	}

	// ��� ����(����) ��ȯ
	return result;
}

/***************************
* SaturationCasting(vector<vector<double>> input -> Mat
* parameter
*	input : �Է� ����(����)
* return : double Ÿ������ ����� ���͸� Mat_<uchar> Ÿ������ ��ȯ�ؼ� opencv���� �̹����� ��� �� �ְ� ������ ������
***************************/
Mat SaturationCasting(vector<vector<double>> input)
{
	Mat castedImage = Mat_<uchar>(input.size(), input[0].size());
	for (int i = 0; i < castedImage.rows; ++i)
		for (int j = 0; j < castedImage.cols; ++j)
			castedImage.data[i * castedImage.cols + j] = saturate_cast<uchar>(input[i][j]);
	return castedImage;
}