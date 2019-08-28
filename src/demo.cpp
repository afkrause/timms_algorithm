#include <iostream>
#include <memory>

 #define OPENCL_ENABLED

#include "timm_two_stage.h"

void main()
{
	using namespace std;

	Timm_two_stage timm;
	

	PRINT_MENU:
	cout << "\n=== Menu Vectorization Level ===\n";
	cout << "[0] no vectorization\n";
	#ifdef _WIN32
	cout << "[1] 128bit SSE OR ARM NEON\n";
	cout << "[2] 256bit AVX2 (default - works on most modern CPUs)\n";
	cout << "[3] 512bit AVX512 (Xeon, Core-X CPUs)\n";
	#endif
	#ifdef __arm__
	cout << "[1] 128bit ARM NEON\n";
	#endif
	#ifdef OPENCL_ENABLED
	cout << "[4] OpenCL\n";
	#endif
	cout << "enter selection:\n";
		
	int sel = 0; cin >> sel;
	switch (sel)
	{
	case 0: timm.setup(USE_NO_VEC); break;
	case 1: timm.setup(USE_VEC128); break;
	case 2: timm.setup(USE_VEC256); break;
	case 3: timm.setup(USE_VEC512); break;
	case 4: timm.setup(USE_OPENCL); break;
	default: cerr << "wrong input. please try again:" << endl; goto PRINT_MENU;
	}

	// select camera
	shared_ptr<cv::VideoCapture> capture;
	while (true)
	{
		cout << "\nplease select a camera id:";
		int cam_nr = 0;
		cin >> cam_nr;
		capture = make_shared<cv::VideoCapture>(cam_nr);
		if (capture->isOpened()) { break; }
		cerr << "\ncould not open and initialize camera nr. " << cam_nr << ". please try again!\n";
	}

	cv::Mat frame, frame_gray;
	cv::Point2f pupil_pos, pupil_pos_coarse;
	while (true)
	{
		capture->read(frame);
		if (!frame.empty())
		{

			// calc pupil center
			cv::cvtColor(frame, frame_gray, cv::COLOR_BGR2GRAY);

			std::tie(pupil_pos, pupil_pos_coarse) = timm.pupil_center(frame_gray);

			timm.visualize_frame(frame, pupil_pos, pupil_pos_coarse);
			cv::imshow("eye_cam", frame);
			cv::waitKey(1);
		}
	}
}


#ifdef OPENCL_ENABLED
#include "opencl_kernel.cpp"
#endif


// libraries + paths (specific for my setup, adjust to your own paths)
#ifdef OPENCL_ENABLED

#pragma comment(lib, "opencl_cu10/lib/x64/opencl.lib")
#endif

#ifdef _DEBUG
#pragma comment(lib, "opencv41/build/x64/vc15/lib/opencv_world411d.lib")
#else
#pragma comment(lib, "opencv41/build/x64/vc15/lib/opencv_world411.lib")
#endif

