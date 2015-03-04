#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <opencv2/imgproc/imgproc.hpp>
#include <math.h>
#include <vector>
#include <fstream>
#include <thread>
#include "HDRWriter.h"

using namespace std;
using namespace cv;


inline void rgb2v(int r, int g, int b, int&v){
	v = r;	// max rgb value
	if (v < g) v = g;
	if (v < b) v = b;
}

/*list.txt contains filenames for images and its exposure time(s)*/
void loadExposureSeq(String path, vector<Mat>& images, vector<float>& times, String gray)
{
	int flag = 1;
	if(!gray.compare("gray"))
		flag = 0;
	path = path + std::string("/");
	ifstream list_file;
	string fileName = path + "list.txt";
	list_file.open(fileName);
	string name;
	float val;
	while (list_file >> name >> val) {
		Mat img = imread(path + name, flag);
		images.push_back(img);
		times.push_back(log(val));
	}
	list_file.close();
}

void splitChannels(vector<Mat>& images, vector<Mat>& Reds, vector<Mat>& Greens, vector<Mat>& Blues){
	for (int i = 0; i < images.size(); i++){
		Mat channels[3];
		split(images[i], channels);
		channels[2].copyTo(Reds[i]);
		channels[1].copyTo(Greens[i]);
		channels[0].copyTo(Blues[i]);
	}
}

void getRandomLocation(Mat values, vector<int>& locations){
	cout << "fuck" << endl;
	int numPixels = values.rows*values.cols;
	locations.clear();
	int sameValues;
	vector<int> index(numPixels);

	uchar* data = values.data;
	for (int i = 0; i < 256; i++){
		sameValues = 0;
		for (int j = 0; j < numPixels; j++){
			if (static_cast<int>(data[j]) == i)
				index[sameValues++] = j;
		}
		if (sameValues){
			int random = rand() % sameValues;
			locations.push_back(index[random]);
		}
	}
}

int weight(int zij){
	int zmin = 0;
	int zmax = 255;
	int zmid = 0.5*(zmax - zmin);
	return zij > zmid ? (zmax - zij) : (zij - zmin);
}


void LSQ(string ch, vector<Mat>* images, vector<float>& times, vector<int>& locations, double lambda,
	vector<float>& lE, vector<float>& g, Mat* map){

	std::cout << "Processing " << ch <<" Channel" << endl;

	Mat A, b, x;
	int sampleSize = locations.size();
	int picNum = (*images).size();
	int n = 256;

	A = Mat::zeros(sampleSize * picNum + 255, n + sampleSize, CV_32F);
	b = Mat::zeros(sampleSize * picNum + 255, 1, CV_32F);
	x = Mat::zeros(n + sampleSize, 1, CV_32F);



	int k = 0;
	for (int i = 0; i < sampleSize; i++){
		int currpos = locations[i];
		for (int j = 0; j < picNum; j++){
			int zij = static_cast<int>((*images)[j].at<uchar>(currpos));
			int wij = weight(zij);
			A.at<float>(k, zij) = static_cast<float>(wij);
			A.at<float>(k, n + i) = -1.0 * static_cast<float>(wij);
			b.at<float>(k) = static_cast<float>(wij*times[j]);
			k++;
		}
	}


	A.at<float>(k, 128) = 1.0;
	k++;

	for (int i = 1; i < 255; i++){
		A.at<float>(k, i - 1) = static_cast<float>(lambda * weight(i));
		A.at<float>(k, i) = static_cast<float>(-2.0 * lambda * weight(i));
		A.at<float>(k, i + 1) = static_cast<float>(-lambda * weight(i));
		k++;
	}
	solve(A, b, x, DECOMP_SVD);


	g.resize(256);
	for (int i = 0; i < 256; i++)
		g[i] = x.at<float>(i);


	//float gmin = g[0];
	//for (int i = 0; i < 50; i++) gmin = gmin < g[i] ? gmin : g[i];
	//for (int i = 0; i < 50; i++) {
	//	if (g[i] == gmin) break;
	//	g[i] = gmin;
	//}

	//float gmax = g[255];
	//for (int i = 255; i > 200; i--) gmax = gmax > g[i] ? gmax : g[i];
	//for (int i = 255; i > 200; i--) {
	//	if (g[i] == gmax) break;
	//	g[i] = gmax;
	//}

	int numFrames = (*images).size();



	for (int ip = 0; ip < (*images)[0].rows; ip++) {
		for (int jp = 0; jp < (*images)[0].cols; jp++){
			float sum_numer = 0.0;
			float sum_denom = 0.0;
			int zij = 0;
			float Ei = 0.f;
			for (int j = 0; j < numFrames; j++) {
				zij = static_cast<int>((*images)[j].at<uchar>(ip, jp));
				if (zij == 255)
					zij = 254;
				if (zij == 0)
					zij = 1;
				float Ei = g[zij] - times[j];
				int wij = weight(zij);
				sum_numer += wij*Ei;
				sum_denom += wij;
			}
			float le = sum_numer / sum_denom;
			(*map).at<float>(ip, jp) = exp(le);
		}
	}

	//if (!ch.compare("blue")){
	//	for (int i = 0; i < images[0].rows / 4; i++)
	//		for (int j = 0; j < images[0].cols / 4; j++){
	//			if (map.at<float>(i, j) > 255.0f)
	//				cout << "While!: " << i << " " << j << endl;
	//		}
	//}
}

void releaseMatVec(vector<Mat>& i){
	for(auto &a : i)
		a.release();
	i.clear();
}

void makehdr(vector<Mat> images, vector<float>& times, double lambda, string output, bool isNIR){
	vector<Mat> Reds(images.size());
	vector<Mat> Greens(images.size());
	vector<Mat> Blues(images.size());

	vector<int> rpixelLocations;
	vector<int> gpixelLocations;
	vector<int> bpixelLocations;

	const int numRows = images[0].rows;
	const int numCols = images[0].cols;
	const int numPixels = numRows*numCols;
	const int channel = images[0].channels();

	if(channel != 1){
		cout << "Color" << endl;
		splitChannels(images, Reds, Greens, Blues);
		if(isNIR){
			for(int i = 0; i < images.size();i++){
				Mat vChanel = Mat::zeros(numRows, numCols, CV_8U);
				for(int r = 0; r < numRows; r++)
					for(int c = 0; c < numCols; c++){
						rgb2v(static_cast<int>(Reds[i].at<uchar>(r,c)),
									static_cast<int>(Greens[i].at<uchar>(r,c)),
									static_cast<int>(Blues[i].at<uchar>(r,c)),
									(int&)vChanel.at<uchar>(r,c));
						Reds[i].at<uchar>(r,c) = vChanel.at<uchar>(r,c);
						Greens[i].at<uchar>(r,c) = vChanel.at<uchar>(r,c);
						Blues[i].at<uchar>(r,c) = vChanel.at<uchar>(r,c);
					}
				vChanel.release();
			}
		}
	}
	else{
		cout << "Gray" << endl;
		for(int i = 0; i < images.size();i++){
			images[i].copyTo(Reds[i]);
			images[i].copyTo(Greens[i]);
			images[i].copyTo(Blues[i]);
		}
	}
	int mid = images.size() / 2;
	//cout << Reds[0].at<float>(0) << endl;
	Mat red_map;
	red_map = Mat::zeros(Reds[0].rows, Reds[0].cols, CV_32F);
	getRandomLocation(Reds[mid], rpixelLocations);
	vector<float> lRE, rg;

	Mat green_map;
	green_map = Mat::zeros(Greens[0].rows, Greens[0].cols, CV_32F);
	getRandomLocation(Greens[mid], gpixelLocations);
	vector<float> lGE, gg;

	Mat blue_map;
	blue_map = Mat::zeros(Blues[0].rows, Blues[0].cols, CV_32F);
	getRandomLocation(Blues[mid], bpixelLocations);
	vector<float> lBE, bg;

	if(channel == 1){
		gpixelLocations = rpixelLocations;
		bpixelLocations = rpixelLocations;
	}

	thread red(LSQ, "red",&Reds, ref(times), ref(rpixelLocations), ref(lambda), ref(lRE), ref(rg), &red_map);
	thread green(LSQ, "green", &Greens, ref(times), ref(gpixelLocations), ref(lambda), ref(lGE), ref(gg), &green_map);
	thread blue(LSQ,"blue", &Blues, ref(times), ref(bpixelLocations), ref(lambda), ref(lBE), ref(bg), &blue_map);

	red.join();
	green.join();
	blue.join();

	cout << "Three channels Done" << endl;

	/* write into hdr file*/

	//float* hdr_data;
	//hdr_data = (float*)malloc(sizeof(float) * (3 * numPixels));
	//for (int i = 0; i < numPixels; i++) {
	//	hdr_data[3 * i + 0] = red_map.at<float>(i);
	//	hdr_data[3 * i + 1] = green_map.at<float>(i);
	//	hdr_data[3 * i + 2] = blue_map.at<float>(i);
	//}

	FILE *fpout = fopen("scene.hdr", "wb");
	assert(fpout != NULL);

	writeRadiance(fpout, red_map, green_map, blue_map, numCols, numRows);
	cout << "Write hdr file done" << endl;
	fclose(fpout);
	/*hdr file done*/

	Mat channels[3] = { blue_map, green_map, red_map};


	imwrite("red.jpeg", red_map);

	imwrite("green.jpeg", green_map);

	imwrite("blue.jpeg", blue_map);

	cout << "Write 3 color file done" << endl;

	Mat oimage;

	merge(channels, 3, oimage);

	imwrite("output.jpeg", oimage);
}

int main(int argc, char** argv){
	vector<Mat> images;
	vector<float> times;
	if(argc < 3){
			cout << "Missing arguments! Need three!" << endl;
			return -1;
	}

	loadExposureSeq(argv[1], images, times,argv[2] );

	bool isNIR = false;
	if(!static_cast<string>(argv[2]).compare("nir"))
		isNIR = true;

	makehdr(images, times, 3.5, "out.hdr", isNIR);
	for(auto &i : images)
		i.release();
	times.clear();
	images.clear();

	return 0;

}
