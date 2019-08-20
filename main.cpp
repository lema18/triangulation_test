#include <stdio.h>
#include <iostream>
#include <ctype.h>
#include <opencv2/opencv.hpp>
#include <cvsba/cvsba.h>
#include <string>
#include <sstream>
#include <stdlib.h>
#include <unistd.h>
#include <ctime>
#include <pcl/common/common_headers.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <Eigen/Core>
#include <Eigen/LU>
#include <Eigen/Geometry>
#include <opencv2/xfeatures2d.hpp>
#include <sys/stat.h>
#include <fstream>
#include <Eigen/StdVector>
#include "DBoW2.h"
#include <algorithm> 
#include <unordered_set>
#include <stdint.h>
#include <algorithm>
#include <iterator> 
#include <vector>
#include "g2o/core/sparse_optimizer.h"
#include "g2o/types/icp/types_icp.h"
#include "g2o/config.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/solvers/dense/linear_solver_dense.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/solvers/structure_only/structure_only_solver.h"
#include "functions.h"
#include <opencv2/core/eigen.hpp>

using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace DBoW2;

int main(int argc,char ** argv)
{
    if(argc<10)
    {
        cout<< "bad input\n";
        cout << "please enter:\n";
        cout << "argv[1]= path to rgb images\n";
        cout << "argv[2]= number of images of the dataset\n";
        cout << "argv[3]= distance ratio to reject features\n";
        cout << "argv[4]= confidence on ransac\n";
        cout << "argv[5]= threshold for error_reprojection on ransac\n";
        cout << "argv[6]= first img index of dataset\n";
        cout << "argv[7]= score threshold for DBoW2\n";
        cout << "argv[8]=path to groundtruth file in txt format\n";
        cout << "argv[9]=path to DBoW2 vocabulary\n";
        exit(-1);
    }
    Mat distcoef = (Mat_<float>(1, 5) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	Mat distor = (Mat_<float>(5, 1) << 0.2624, -0.9531, -0.0054, 0.0026, 1.1633);
	distor.convertTo(distor, CV_64F);
	Mat intrinsic = (Mat_<float>(3, 3) << 516.9, 0., 318.6, 0., 516.9, 255.3, 0., 0., 1.);
    //fx=517.3 // fy=516.5
	intrinsic.convertTo(intrinsic, CV_64F);
	distcoef.convertTo(distcoef, CV_64F);
    double focal_length=516.9;
    Point2d principal_point(318.6,255.3); 
    //number of images to compose the initial map
    int nImages =  atoi(argv[2]);
    //threshold for points tha are seen in more than i images
    //distance ratio to reject features
    double dst_ratio;
    sscanf(argv[3],"%lf",&dst_ratio);
    //confidence on ransac
    double confidence;
    sscanf(argv[4],"%lf",&confidence);
    //threshold for error_reprojection on ransac
    double reproject_err;
    sscanf(argv[5],"%lf",&reproject_err);
    //dataset_offset
    int offset;
    sscanf(argv[6],"%d",&offset);
    double dbow2_threshold;
    sscanf(argv[7],"%lf",&dbow2_threshold);
    //first index of dataset
    int current_frame,last_frame;
    last_frame=offset;
    current_frame=last_frame;
    //ident for 3d_points
    //to store index of keyframes from dataset
    vector<int> keyframes;
    //to store used features
    vector<int> aux_inliers;
    //to store camera poses
    //my_slam obj;
    //initialization
    double score=1;// DBoW2 score
    vector<cv::Mat> camera_poses;
    

    //-------------initialization---------------------------
    //we read the first keyframe
    my_slam frames_info;
    frames_info.imgs[last_frame]=loadImage(argv[1],last_frame,intrinsic,distcoef);
    vector<KeyPoint> features1;
    cv::Mat descriptors1;
    calculate_features_and_descriptors(frames_info.imgs[last_frame],features1,descriptors1);
    frames_info.features[last_frame]=features1;
    frames_info.descriptors[last_frame]=descriptors1;
    frames_info.valid_frames.push_back(1);
    frames_info.frames_id.push_back(last_frame);
    frames_info.cam_poses[last_frame]=cv::Mat::eye(4,4,CV_64F);
    //usind DBoW2 vocabulary we search next keyframe
    int last_found=0;
    while(current_frame<nImages || !last_found)
    {
        current_frame++;
        vector<KeyPoint> features2;
        cv::Mat descriptors2;
        frames_info.imgs[current_frame]=loadImage(argv[1],current_frame,intrinsic,distcoef);
        calculate_features_and_descriptors(frames_info.imgs[current_frame],features2,descriptors2);
        frames_info.descriptors[current_frame]=descriptors2;
        frames_info.features[current_frame]=features2;
        vector<cv::Mat> left_descriptors,right_descriptors;
        changeStructure(frames_info.descriptors[last_frame],left_descriptors);
        changeStructure(frames_info.descriptors[current_frame],right_descriptors);
        score=calculate_score(argv[9],last_frame,current_frame,left_descriptors,right_descriptors);
        if(score<=dbow2_threshold)
        {
            frames_info.valid_frames.push_back(1);
            frames_info.frames_id.push_back(current_frame);
            //keyframe found lets match features between them
            vector<Point2f> left_points,right_points;
            vector<int> left_idx,right_idx;
            cv::Mat mask;
            cv::Mat E;
            matchFeatures_and_compute_essential(last_frame,current_frame,left_points,right_points,left_idx,right_idx,mask,dst_ratio,
            E,focal_length,confidence,reproject_err,principal_point,frames_info);
            double scale=1;
            vector<Point3d> triangulated_points;
            estimate_motion_and_calculate_3d_points(last_frame,current_frame,left_points,right_points,left_idx,right_idx,
            intrinsic,E,focal_length,principal_point,scale,mask,triangulated_points,frames_info);
            score=1;
            last_frame=current_frame;
            last_found=1;
        }
        else
        {
            frames_info.valid_frames.push_back(0);
            frames_info.frames_id.push_back(current_frame);
            last_found=0;
        }
    }
    //visualization
    pcl::visualization::PCLVisualizer viewer("Viewer");
	viewer.setBackgroundColor(0.35, 0.35, 0.35); 

    mkdir("./lectura_datos", 0777);
    std::ofstream file("./lectura_datos/odometry.txt");
    if (!file.is_open()) return -1;

	for (unsigned int i = 0; i < frames_info.valid_frames.size(); i++)
    {
        if(frames_info.valid_frames[i]==1)
        {
            stringstream sss;
		    string name;
		    sss << frames_info.frames_id[i];
		    name = sss.str();
		    Eigen::Affine3f cam_pos;
            Eigen::Matrix4d eig_cam_pos=Eigen::Matrix4d::Identity();
            Eigen::Vector3d cam_translation;
            Eigen::Matrix3d cam_rotation;
            cv::Mat cv_cam_rot;
            cv::Mat cv_cam_tras;
            cv_cam_rot=frames_info.cam_poses[i].rowRange(0,3).colRange(0,3);
            cv_cam_tras=frames_info.cam_poses[i].rowRange(0,3).col(3);
            cv2eigen(cv_cam_rot,cam_rotation);
            cv2eigen(cv_cam_tras,cam_translation);
            eig_cam_pos.block<3,3>(0,0)=cam_rotation;
            eig_cam_pos.block<3,1>(0,3) = cam_translation;
            cam_pos=eig_cam_pos.cast<float>();
            viewer.addCoordinateSystem(1, cam_pos, name);
		    pcl::PointXYZ textPoint(cam_pos(0,3), cam_pos(1,3), cam_pos(2,3));
		    viewer.addText3D(std::to_string(i), textPoint, 0.1, 1, 1, 1, "text_"+std::to_string(i));
            Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));

            file << frames_info.frames_id[i] << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
                q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
        }
    }
    file.close();
    read_poses_from_ground_truth(argv[8],((int)frames_info.valid_frames.size())*4,camera_poses);
    generate_ground_truth_for_comparation(camera_poses);
    
	pcl::PointCloud<pcl::PointXYZ> cloud;
    for(int j=0;j<frames_info.ident;j++)
    {
        pcl::PointXYZ p(frames_info.triangulated_points[j].x, frames_info.triangulated_points[j].y,frames_info.triangulated_points[j].z);
        cloud.push_back(p);
    }
	viewer.addPointCloud<pcl::PointXYZ>(cloud.makeShared(), "map");

	while (!viewer.wasStopped()) {
		viewer.spin();
	}
	return 0;

}
