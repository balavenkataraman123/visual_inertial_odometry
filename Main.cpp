#pragma once
#include <depthai/depthai.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/quaternion.hpp>
#include <future>

/////////	|
// CONFIG	V

#define CfgFirstFrame		100
#define CfgImageSize		160
#define CfgMaxQueueSize		3
#define CfgMaxPts			20
#define CfgAlphaVIO			0.1

/////////	|
// DRIVER	V

std::shared_ptr<dai::Pipeline> Pipeline; std::shared_ptr<dai::Device> Device;
std::shared_ptr<dai::node::MonoCamera> LeftCamera; std::shared_ptr<dai::node::XLinkOut> LeftCamera_Link;
std::shared_ptr<dai::node::MonoCamera> RightCamera; std::shared_ptr<dai::node::XLinkOut> RightCamera_Link;
std::shared_ptr<dai::node::IMU> IMU; std::shared_ptr<dai::node::XLinkOut> IMU_Link;
dai::CalibrationHandler Calibration;

void get_camera_intrinsics(cv::Mat& Intrinsics)
{
	Pipeline = std::make_shared<dai::Pipeline>();

	int CamFPS = 20;
	int RotFPS = 60;

	dai::MonoCameraProperties::SensorResolution SensorResolution = dai::MonoCameraProperties::SensorResolution::THE_400_P;

	LeftCamera = Pipeline->create<dai::node::MonoCamera>(); LeftCamera->setBoardSocket(dai::CameraBoardSocket::CAM_B);
	LeftCamera->setResolution(SensorResolution); LeftCamera->setFps(CamFPS);
	LeftCamera_Link = Pipeline->create<dai::node::XLinkOut>(); LeftCamera_Link->setStreamName("LeftCamera");
	LeftCamera->out.link(LeftCamera_Link->input);

	RightCamera = Pipeline->create<dai::node::MonoCamera>(); RightCamera->setBoardSocket(dai::CameraBoardSocket::CAM_C);
	RightCamera->setResolution(SensorResolution); RightCamera->setFps(CamFPS);
	RightCamera_Link = Pipeline->create<dai::node::XLinkOut>(); RightCamera_Link->setStreamName("RightCamera");
	RightCamera->out.link(RightCamera_Link->input);

	IMU = Pipeline->create<dai::node::IMU>(); IMU->enableIMUSensor(dai::IMUSensor::ARVR_STABILIZED_GAME_ROTATION_VECTOR, RotFPS);
	IMU->setBatchReportThreshold(1); IMU->setMaxBatchReports(1); IMU_Link = Pipeline->create<dai::node::XLinkOut>();
	IMU_Link->setStreamName("IMU"); IMU->out.link(IMU_Link->input);

	Device = std::make_shared<dai::Device>(*(Pipeline), dai::UsbSpeed::HIGH); Calibration = Device->readCalibration();
	Intrinsics = cv::Mat::eye(3, 3, CV_64F); auto m = Calibration.getCameraIntrinsics(dai::CameraBoardSocket::CAM_B, CfgImageSize, CfgImageSize);
	Intrinsics.at<double>(0, 0) = m[0][0];
	Intrinsics.at<double>(1, 1) = m[1][1];
	Intrinsics.at<double>(0, 2) = m[0][2];
	Intrinsics.at<double>(1, 2) = m[1][2];
}

struct StereoPair {
	cv::Mat LeftImage;
	cv::Mat RightImage;
	cv::Vec3d rvec;
	double fx;
	double fy;
	double Baseline;
};

void read_image(StereoPair& Image)
{
	Image = StereoPair();
	static cv::Mat Intrinsics; if (Intrinsics.empty()) get_camera_intrinsics(Intrinsics);

	Image.LeftImage = Device->getOutputQueue("LeftCamera", CfgMaxQueueSize, false)->get<dai::ImgFrame>()->getCvFrame();
	cv::resize(Image.LeftImage, Image.LeftImage, cv::Size(CfgImageSize, CfgImageSize));

	Image.RightImage = Device->getOutputQueue("RightCamera", CfgMaxQueueSize, false)->get<dai::ImgFrame>()->getCvFrame();
	cv::resize(Image.RightImage, Image.RightImage, cv::Size(CfgImageSize, CfgImageSize));

	Image.fx = Intrinsics.at<double>(0, 0) / CfgImageSize;
	Image.fy = Intrinsics.at<double>(1, 1) / CfgImageSize;

	Image.Baseline = Calibration.getBaselineDistance();

	auto IMU = Device->getOutputQueue("IMU", CfgMaxQueueSize, false)->get<dai::IMUData>()->packets.back().rotationVector;
	cv::Mat R = (cv::Mat)cv::Quatd(IMU.k, -IMU.i, -IMU.j, IMU.real).toRotMat3x3(cv::QuatAssumeType::QUAT_ASSUME_NOT_UNIT);
	R.convertTo(R, CV_64F);
	static cv::Mat O = R.clone();
	R *= O.inv();
	cv::Rodrigues(R, Image.rvec);
}

//////////	|
// TRACKER	V

cv::Mat get_intrinsics(StereoPair Image);
cv::Vec3d from_images_to_fast_translation(StereoPair ImageA, StereoPair ImageB, int MaxPts=CfgMaxPts);

template <typename WorkerDataIn, typename WorkerDataOut>
class WorkerThread 
{
public:
	virtual void worker_setup(WorkerDataIn& InputData) { }
	virtual WorkerDataOut worker_main(WorkerDataIn InputData) { return WorkerDataOut(); }

private:
	std::future<WorkerDataOut> WorkerTask;
	bool IsInit = true;
	bool IsComplete = false;

public:
	WorkerDataOut _worker(WorkerDataIn InputData) 
	{
		IsComplete = false;
		WorkerDataOut OutputData = worker_main(InputData);
		IsComplete = true;
		return OutputData;
	}

	void launch_worker(WorkerDataIn InputData) 
	{
		if (IsInit || IsComplete) {
			IsInit = false;
			worker_setup(InputData);
			WorkerTask = std::async(std::launch::async, [&](WorkerDataIn In) { return _worker(In); }, InputData);
		}
	}

	bool try_get_output(WorkerDataOut& OutputData) {
		if (IsComplete) {
			OutputData = WorkerTask.get();
			return true;
		}
		return false;
	}
};

struct VIO_Input {
	StereoPair Prev;
	StereoPair Curr;
};

struct VIO_Output {
	cv::Vec3d TRel;
};

class VIO_Worker : public WorkerThread<VIO_Input, VIO_Output>
{
public:
	virtual void worker_setup(VIO_Input& InputData)
	{
		InputData.Prev.LeftImage = InputData.Prev.LeftImage.clone();
		InputData.Prev.RightImage = InputData.Prev.RightImage.clone();

		InputData.Curr.LeftImage = InputData.Curr.LeftImage.clone();
		InputData.Curr.RightImage = InputData.Curr.RightImage.clone();
	}

	cv::Mat get_rglb(StereoPair Image) 
	{
		cv::Mat RGlb;
		cv::Rodrigues(Image.rvec, RGlb);
		RGlb = RGlb.inv();
		return RGlb.clone();
	}

	cv::Vec3d get_rvec(StereoPair Image) 
	{
		cv::Mat RGlb = get_rglb(Image);
		cv::Vec3d rvec;
		cv::Rodrigues(RGlb, rvec);
		return rvec;
	}

	virtual VIO_Output worker_main(VIO_Input InputData)
	{ 
		VIO_Output OutputData;

		cv::Vec3d TRel = from_images_to_fast_translation(InputData.Prev, InputData.Curr);
		cv::Mat RGlb = get_rglb(InputData.Curr);
		TRel = (cv::Vec3d)(cv::Mat)(RGlb * TRel);

		OutputData.TRel = TRel;

		return OutputData;
	}
};

int main()
{
	cv::Vec3d TGlb(0, 0, 0);
	cv::Vec3d TGlbNoisy(0, 0, 0);

	std::future<cv::Vec3d> VIO_Task;

	StereoPair CurrentImage;
	StereoPair PreviousImage;

	for (int i = 0; i < CfgFirstFrame; i++) { StereoPair _; read_image(_); }
	while (true)
	{
		clock_t Start = clock();

		read_image(CurrentImage);
		if (CurrentImage.LeftImage.channels() == 1) cv::cvtColor(CurrentImage.LeftImage, CurrentImage.LeftImage, cv::COLOR_GRAY2BGR);
		if (CurrentImage.RightImage.channels() == 1) cv::cvtColor(CurrentImage.RightImage, CurrentImage.RightImage, cv::COLOR_GRAY2BGR);

		if (PreviousImage.LeftImage.empty()) {
			PreviousImage = CurrentImage;
			PreviousImage.LeftImage = CurrentImage.LeftImage.clone();
			PreviousImage.RightImage = CurrentImage.RightImage.clone();
		}

		cv::Mat Intrinsics = get_intrinsics(CurrentImage);

		static VIO_Worker VIO_WorkerTask;
		VIO_WorkerTask.launch_worker({ PreviousImage, CurrentImage });

		VIO_Output VIO_OutputData;
		if (VIO_WorkerTask.try_get_output(VIO_OutputData)) {
			TGlbNoisy += VIO_OutputData.TRel;

			PreviousImage = CurrentImage;
			PreviousImage.LeftImage = CurrentImage.LeftImage.clone();
			PreviousImage.RightImage = CurrentImage.RightImage.clone();
		}

		TGlb = TGlbNoisy * (1.0 - CfgAlphaVIO) + TGlb * CfgAlphaVIO;

		cv::Mat FinalRender = cv::Mat::zeros(CurrentImage.LeftImage.size(), CV_8UC3);
		cv::Vec3d rvec = VIO_WorkerTask.get_rvec(CurrentImage);
		cv::drawFrameAxes(FinalRender, Intrinsics, cv::Mat(), rvec, TGlb, 15.24, 15);
		cv::imshow("Game", FinalRender);
		cv::waitKey(1);

		cv::Mat RGlb = VIO_WorkerTask.get_rglb(CurrentImage);
		cv::Vec3d Rot1 = cv::Quatd::createFromRotMat(RGlb).toRotVec();
		cv::Quatd Rot2 = cv::Quatd::createFromRvec(cv::Vec3d(Rot1[0] * -1, Rot1[1], Rot1[2] * -1));

		cv::Vec3f CameraPosition(TGlb[0], TGlb[1] * -1, TGlb[2]);
		cv::Vec4f CameraRotation(Rot2.x, Rot2.y, Rot2.z, Rot2.w);

		std::cout << CameraPosition << CameraRotation << std::endl;
		// TODO: DON'T SEND THROUGH CONSOLE
		// TODO: TRY FOR ALL CAMERAS ON HMD TO FIND THE BEST TRANSFORM
	}
}

//////	|
// AUX	V

cv::Mat get_intrinsics(StereoPair Image) {
	cv::Mat Intrinsics = cv::Mat::eye(3, 3, CV_64F);
	Intrinsics.at<double>(0, 0) = Image.fx * Image.LeftImage.cols;
	Intrinsics.at<double>(1, 1) = Image.fy * Image.LeftImage.rows;
	Intrinsics.at<double>(0, 2) = Image.LeftImage.cols / 2.0;
	Intrinsics.at<double>(1, 2) = Image.LeftImage.rows / 2.0;
	return Intrinsics;
}

StereoPair unrotate(StereoPair ImageA, StereoPair ImageB)
{
	StereoPair UnrotatedImageA = ImageA;
	UnrotatedImageA.LeftImage = ImageA.LeftImage.clone();
	UnrotatedImageA.RightImage = ImageA.RightImage.clone();

	cv::Mat Intrinsics = get_intrinsics(ImageA);

	cv::Mat R1; cv::Rodrigues(ImageA.rvec, R1); R1.convertTo(R1, CV_64F);
	cv::Mat R2; cv::Rodrigues(ImageB.rvec, R2); R2.convertTo(R2, CV_64F);
	cv::Mat R = R2 * R1.inv();
	R.convertTo(R, CV_64F);
	R = Intrinsics * R * Intrinsics.inv();

	cv::warpPerspective(ImageA.LeftImage, UnrotatedImageA.LeftImage, R, ImageA.LeftImage.size());
	cv::warpPerspective(ImageA.RightImage, UnrotatedImageA.RightImage, R, ImageA.RightImage.size());
	return UnrotatedImageA;
}

cv::Mat from_images_to_projection(StereoPair ImageA, StereoPair ImageB, int MaxPts)
{
	std::vector<cv::Point2f> PtsA;
	std::vector<cv::Point2f> PtsB;
	std::vector<cv::KeyPoint> Kps;
	cv::Ptr<cv::SIFT> SIFT = cv::SIFT::create(MaxPts);
	SIFT->detect(ImageA.LeftImage, Kps);
	if (!Kps.empty()) {
		cv::Mat st, err;
		for (cv::KeyPoint kp : Kps) PtsA.push_back(kp.pt);
		cv::calcOpticalFlowPyrLK(ImageA.LeftImage, ImageB.LeftImage, PtsA, PtsB, st, err);
		for (int i = st.total() - 1; i >= 0; i--) {
			if (st.at<uchar>(i) != 1) {
				PtsA.erase(PtsA.begin() + i);
				PtsB.erase(PtsB.begin() + i);
			}
		}
	}
	if (PtsA.size() >= 3) {
		cv::Mat T = cv::estimateAffinePartial2D(PtsA, PtsB);
		T.convertTo(T, CV_64F);
		cv::vconcat(T, cv::Vec3d(0, 0, 1).t(), T);
		return T.clone();
	}
	return cv::Mat();
}

cv::Vec3d from_images_to_direction(StereoPair ImageA, StereoPair ImageB, int MaxPts, double& PixelsMoved)
{
	cv::Mat T = from_images_to_projection(ImageA, ImageB, MaxPts);
	if (T.empty()) return cv::Vec3d(0, 0, 0);
	cv::Mat Intrinsics = get_intrinsics(ImageA);
	std::vector<cv::Point3d> Obj = {
		cv::Point3d(0, ImageA.LeftImage.rows, 0),
		cv::Point3d(ImageA.LeftImage.cols, ImageA.LeftImage.rows, 0),
		cv::Point3d(ImageA.LeftImage.cols, 0, 0),
		cv::Point3d(0, 0, 0) };
	std::vector<cv::Point2d> Src = {
		cv::Point2d(0, 0),
		cv::Point2d(ImageA.LeftImage.cols, 0),
		cv::Point2d(ImageA.LeftImage.cols, ImageA.LeftImage.rows),
		cv::Point2d(0, ImageA.LeftImage.rows) };
	std::vector<cv::Point2d> Dst;
	cv::perspectiveTransform(Src, Dst, T);
	cv::Vec3d tvec, _;
	cv::solvePnP(Obj, Dst, Intrinsics, cv::Mat(), _, tvec);
	static cv::Vec3d O = tvec;
	tvec -= O;
	PixelsMoved = cv::norm(tvec);
	if (PixelsMoved > 0) tvec /= PixelsMoved;
	tvec *= -1;
	return tvec;
}

double get_absolute_scale(StereoPair ImageA, StereoPair ImageB, int MaxPts, double PixelsMoved)
{
	StereoPair MonoLeft2;
	StereoPair MonoRight2;
	MonoLeft2 = ImageB;
	MonoRight2 = ImageB;
	MonoLeft2.LeftImage = ImageB.LeftImage.clone();
	MonoLeft2.RightImage = MonoLeft2.LeftImage.clone();
	MonoRight2.LeftImage = ImageB.RightImage.clone();
	MonoRight2.RightImage = MonoRight2.LeftImage.clone();
	double Disparity = 0;
	from_images_to_direction(MonoLeft2, MonoRight2, MaxPts, Disparity);
	cv::Mat Intrinsics = get_intrinsics(ImageA);
	double Depth = ImageA.Baseline * Intrinsics.at<double>(0, 0) / Disparity;
	double Scale = ((PixelsMoved / Intrinsics.at<double>(0, 0)) * Depth) * 2.0;
	if (!std::isnormal(Scale)) Scale = 0;
	return Scale;
}

cv::Vec3d from_images_to_fast_translation(StereoPair ImageA, StereoPair ImageB, int MaxPts)
{
	cv::Mat Intrinsics = get_intrinsics(ImageA);
	double Baseline = ImageA.Baseline;

	ImageA = unrotate(ImageA, ImageB);
	double PixelsMoved;
	cv::Vec3d Direction = from_images_to_direction(ImageA, ImageB, MaxPts, PixelsMoved);

	double Delta = get_absolute_scale(ImageA, ImageB, MaxPts, PixelsMoved);

	return Direction * Delta;
}