#ifndef _HDRWRITER_
#define _HDRWRITER_

#include <vector>
#include <cstdio>
#include <cmath>
#include <opencv2/core/core.hpp>

using namespace std;

#define WHITE_EFFICACY 179.0f

typedef unsigned char Trgbe;
struct Trgbe_pixel
{
	Trgbe r;
	Trgbe g;
	Trgbe b;
	Trgbe e;
};


int RLEWrite(FILE* file, Trgbe* scanline, int size)
{
	Trgbe* scanend = scanline + size;
	while (scanline < scanend)
	{
		int run_start = 0;
		int peek = 0;
		int run_len = 0;
		while (run_len <= 4 && peek < 128 && ((scanline + peek) < scanend))
		{
			run_start = peek;
			run_len = 0;
			while ((run_len < 127) &&
				(run_start + run_len < 128) &&
				(scanline + peek < scanend) &&
				(scanline[run_start] == scanline[peek]))
			{
				peek++;
				run_len++;
			}
		}

		if (run_len > 4)
		{
			// write a non run: scanline[0] to scanline[run_start]
			if (run_start > 0)
			{
				std::vector<Trgbe> buf(run_start + 1);

				buf[0] = run_start;
				for (int i = 0; i < run_start; i++) {
					buf[i + 1] = scanline[i];
				}
				fwrite(buf.data(), sizeof(Trgbe), run_start + 1, file);
			}

			// write a run: scanline[run_start], run_len
			Trgbe buf[2];
			buf[0] = 128 + run_len;
			buf[1] = scanline[run_start];
			fwrite(buf, sizeof(*buf), 2, file);
		}
		else
		{
			// write a non run: scanline[0] to scanline[peek]
			std::vector<Trgbe> buf(peek + 1);

			buf[0] = peek;
			for (int i = 0; i<peek; i++) {
				buf[i + 1] = scanline[i];
			}
			fwrite(buf.data(), sizeof(Trgbe), peek + 1, file);
		}
		scanline += peek;
	}

	//if (scanline != scanend)
	//{
	//	throw pfs::io::WriteException("RGBE: difference in size while writing RLE scanline");
	//}

	return 0;
}



void rgb2rgbe(float r, float g, float b, Trgbe_pixel& rgbe)
{
	r /= WHITE_EFFICACY;
	g /= WHITE_EFFICACY;
	b /= WHITE_EFFICACY;

	double v = r;	// max rgb value
	if (v < g) v = g;
	if (v < b) v = b;

	if (v < 1e-32) {
		rgbe.r = rgbe.g = rgbe.b = rgbe.e = 0;
	}
	else {
		int e;	// exponent

		v = frexp(v, &e) * 256.0 / v;
		rgbe.r = Trgbe(v*r);
		rgbe.g = Trgbe(v*g);
		rgbe.b = Trgbe(v*b);
		rgbe.e = Trgbe(e + 128);
	}
}

void writeRadiance(FILE *file,
	const cv::Mat& r, const cv::Mat& g, const cv::Mat& b, int iwidth, int iheight)
{
	size_t width = iwidth;
	size_t height = iheight;

	// DEBUG_STR << "RGBE: writing image " << width << "x" << height << endl;

	// header information
	fprintf(file, "#?RADIANCE\n");	// file format specifier
	fprintf(file, "# Qile Zhu\n");

	// if ( exposure_isset )
	//      fprintf(file, "EXPOSURE=%f\n", exposure);
	// if ( gamma_isset )
	//      fprintf(file, "GAMMA=%f\n", gamma);

	fprintf(file, "FORMAT=32-bit_rle_rgbe\n");
	fprintf(file, "\n");

	// image size
	fprintf(file, "-Y %d +X %d\n", (int)height, (int)width);

	// image run length encoded
	std::vector<Trgbe> scanlineR(width);
	std::vector<Trgbe> scanlineG(width);
	std::vector<Trgbe> scanlineB(width);
	std::vector<Trgbe> scanlineE(width);

	for (size_t y = 0; y<height; ++y)
	{
		// write rle header
		unsigned char header[4];
		header[0] = 2;
		header[1] = 2;
		header[2] = width >> 8;;
		header[3] = width & 0xFF;
		fwrite(header, sizeof(header), 1, file);

		// each channel is encoded separately
		for (size_t x = 0; x < width; x++)
		{
			Trgbe_pixel p;
			rgb2rgbe(r.at<float>(y, x), g.at<float>(y, x), b.at<float>(y, x), p);
			scanlineR[x] = p.r;
			scanlineG[x] = p.g;
			scanlineB[x] = p.b;
			scanlineE[x] = p.e;
		}
		RLEWrite(file, scanlineR.data(), width);
		RLEWrite(file, scanlineG.data(), width);
		RLEWrite(file, scanlineB.data(), width);
		RLEWrite(file, scanlineE.data(), width);
	}
	scanlineR.clear();
	scanlineG.clear();
	scanlineB.clear();
	scanlineE.clear();

}


#endif
