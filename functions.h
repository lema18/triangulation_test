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
#include <opencv2/core/eigen.hpp>


using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
using namespace DBoW2;

class my_slam
{
    public:
    //typedef std::shared_ptr<my_slam> Ptr;
    unordered_map<int,cv::Mat> imgs;
    unordered_map<int,vector<cv::KeyPoint>> features;
    unordered_map<int,cv::Mat> descriptors;
    unordered_map<int,cv::Mat> cam_poses;
    unordered_map<int,vector<Point2f>> projections_of_3d_points;//projections of a single 3d points in each image where it's seen
    unordered_map<int,vector<int>> visibility_of_3d_points;//images where a single point is seen
    unordered_map<int,int> index_of_last_match_descriptor;//descriptor_idx
    unordered_map<int,cv::Point3d> triangulated_points;
    int ident=0;//identifier for 3dpoint
    vector<int> frames_id;
    vector<int> valid_frames;
};
void add_new_3d_point(int last_frame,int current_frame,cv::Point3d pt,int last_match,int curr_match,my_slam &obj)
{
    int ident=obj.ident;
    obj.triangulated_points[ident]=pt;
    obj.visibility_of_3d_points[ident].push_back(last_frame);
    obj.visibility_of_3d_points[ident].push_back(current_frame);
    obj.projections_of_3d_points[ident].push_back(obj.features[last_frame][last_match].pt);
    obj.projections_of_3d_points[ident].push_back(obj.features[current_frame][curr_match].pt);
    obj.index_of_last_match_descriptor[ident]=curr_match;
    obj.ident++;
}
void search_existent_points(int ident,vector<int> ref_match_idx,vector<int> &used_points,vector<int> &identifiers,my_slam obj)
{
    for(unsigned int i=0;i<ref_match_idx.size();i++)
    {   
        int stop_flag=0;
        for(int j=0;j<ident && !stop_flag;j++)
        {
            if(obj.index_of_last_match_descriptor[j]==ref_match_idx[i])
            {
                used_points.push_back(1);
                identifiers.push_back(j);
                stop_flag=1;
            }
        }
        if(!stop_flag)
        {
            used_points.push_back(0);
            identifiers.push_back(0);
        }   
    }
}
void update_3d_point(my_slam &obj,int identifier,int match_id,int current_frame,cv::Point3d pt)
{
    obj.triangulated_points[identifier]+=pt;
    obj.triangulated_points[identifier]/=2;
    obj.index_of_last_match_descriptor[identifier]=match_id;
    obj.projections_of_3d_points[identifier].push_back(obj.features[current_frame][match_id].pt);
    obj.visibility_of_3d_points[identifier].push_back(current_frame);
}
void pixel_to_cam_plane(Point2d &pixel_plane,Point2d &cam_plane,cv::Mat &intrinsic)
{
    cam_plane.x=(pixel_plane.x-intrinsic.at<double>(0,2))/intrinsic.at<double>(0,0);
    cam_plane.y=(pixel_plane.y-intrinsic.at<double>(1,2))/intrinsic.at<double>(1,1);
}
cv::Mat generate_projection_matrix(cv::Mat &R,cv::Mat &t)
{
    cv::Mat inter;
    hconcat(R,t,inter);
    return inter;
}
void relative_triangulation(vector<Point2d> triang_left,vector<Point2d> triang_right,cv::Mat intrinsic,
cv::Mat R_2_to_1,cv::Mat t_2_to_1,vector<Point3d> &pts3d)
{
    vector<Point2d> normalized_left,normalized_right;
    for(unsigned int i=0;i<triang_left.size();i++)
    {
        cv::Point2d left,right;
        pixel_to_cam_plane(triang_left[i],left,intrinsic);
        normalized_left.push_back(left);
        pixel_to_cam_plane(triang_right[i],right,intrinsic);
        normalized_right.push_back(right);
    }
    cv::Mat left_project_mat=cv::Mat::eye(3,4,CV_64F);
    cout<< "left_project_mat= "<<endl<<" "<< left_project_mat <<endl << endl;
    cv::Mat right_project_mat=generate_projection_matrix(R_2_to_1,t_2_to_1);
    cout<< "right_project_mat= "<<endl<<" "<< right_project_mat <<endl << endl;
    cv::Mat point3d_homo;
    triangulatePoints(left_project_mat,right_project_mat,normalized_left,normalized_right,point3d_homo);
    pts3d.clear();
    for( int i=0;i<point3d_homo.cols;i++)
    {
        Point3d aux;
        aux.x=(point3d_homo.col(i).at<double>(0)/point3d_homo.col(i).at<double>(3));
        aux.y=(point3d_homo.col(i).at<double>(1)/point3d_homo.col(i).at<double>(3));
        aux.z=(point3d_homo.col(i).at<double>(2)/point3d_homo.col(i).at<double>(3));
        pts3d.push_back(aux);
    }
}


void rel_motion_2_to_1(const cv::Mat &ref_cam_pose,cv::Mat &curr_cam_pos,cv::Mat &rel_curr_to_ref)
{
    rel_curr_to_ref=curr_cam_pos.inv()*ref_cam_pose;
}
void update_scale(vector<Point3d> existing,vector<Point3d> corresponding,double &scale)
  {
    vector<double> scales;
    for (size_t j=0; j < existing.size()-1; j++)
    {
      for (size_t k=j+1; k< existing.size(); k++)
      {
        double s = norm(existing[j] - existing[k]) / norm(corresponding[j] - corresponding[k]);
        scales.push_back(s);
      }
    }
    sort(scales.begin(),scales.end());
    int n=scales.size();
    if (n % 2 != 0) scale=scales[n/2];
    else scale=(scales[(n-1)/2] + scales[n/2])/2.0;
}
void displayMatches(cv::Mat &_img1, std::vector<cv::Point2d> &_features1,cv::Mat &_img2, std::vector<cv::Point2d> &_features2)
{
	cv::Mat display;
	cv::hconcat(_img1, _img2, display);
	cv::cvtColor(display, display, CV_GRAY2BGR);
    for(unsigned int i = 0; i < _features1.size(); i++)
    {
		auto p1 = _features1[i];
		auto p2 = _features2[i] + cv::Point2d(_img1.cols, 0);
		cv::circle(display, p1, 2, cv::Scalar(0,255,0),2);
		cv::circle(display, p2, 2, cv::Scalar(0,255,0),2);
		cv::line(display,p1, p2, cv::Scalar(0,255,0),1);
	}
    cv::imshow("display", display);
	cv::waitKey(3);
}

void find_last_valid_idx(int frame,int &valid_id,my_slam obj)
{
    for (int i=0;i<frame;i++)
    {
        if(obj.valid_frames[i]==1)
        {
            valid_id=obj.frames_id[i];
        }
    }
}
cv::Mat generate_4x4_transformation(cv::Mat &R,cv::Mat &t)
{
    Mat T = (Mat_<double>(4, 4) << R.at<double>(0, 0), R.at<double>(0, 1), R.at<double>(0, 2), t.at<double>(0, 0),
             R.at<double>(1, 0), R.at<double>(1, 1), R.at<double>(1, 2), t.at<double>(1, 0),
             R.at<double>(2, 0), R.at<double>(2, 1), R.at<double>(2, 2), t.at<double>(2, 0),
             0, 0, 0, 1);
    return T;
}
cv::Mat loadImage(std::string _folder, int _number, cv::Mat &_intrinsics, cv::Mat &_coeffs)
{
	stringstream ss; /*convertimos el entero l a string para poder establecerlo como nombre de la captura*/
	ss << _folder << "/left_" << _number << ".png";
	std::cout << "Loading image: " << ss.str() << std::endl;
	Mat image = imread(ss.str(), CV_LOAD_IMAGE_COLOR);
    cv::Mat image_u;
	undistort(image, image_u, _intrinsics, _coeffs);
    cvtColor(image_u, image_u, COLOR_BGR2GRAY);
	return image_u;
}
void changeStructure(const cv::Mat &plain, vector<cv::Mat> &out)
{
    out.resize(plain.rows);
    for(int i = 0; i < plain.rows; ++i)
    {
        out[i] = plain.row(i);
    }
}
cv::Point3d change_points_to_other_ref_system(const cv::Point3d &old_pt,cv::Mat &curr_pos)
{
    const Mat &T = curr_pos;
    double p[4] = {old_pt.x, old_pt.y, old_pt.z, 1};
    double res[3] = {0, 0, 0};
    for (int row = 0; row < 3; row++)
    {
        for (int j = 0; j < 4; j++)
            res[row] += T.at<double>(row, j) * p[j];
    }
    return Point3d(res[0], res[1], res[2]);
}
double calculate_score(string path,int last_frame,int current_frame,
                        const vector<cv::Mat> &descriptors_left,const vector<cv::Mat> &descriptors_right)
{
    // lets do something with this vocabulary
    // load the vocabulary from disk
    OrbVocabulary voc(path);
    cout << "Matching images against themselves (0 low, 1 high): " << endl;
    BowVector v1, v2;
    voc.transform(descriptors_left, v1);
    voc.transform(descriptors_right, v2);
    double score = voc.score(v1, v2);
    cout << "Image " << last_frame << " vs Image " << current_frame << ": " << score << endl;
    return score;
}
void calculate_features_and_descriptors(cv::Mat &img,vector<KeyPoint> &features,cv::Mat &descriptors)
{
    auto pt=ORB::create();
    pt->detectAndCompute(img,cv::Mat(),features,descriptors);    
}
void matchFeatures_and_compute_essential(int last_frame,int current_frame,
                                    vector<Point2f> &corresponding_left,vector<Point2f> &corresponding_right,
                                    vector<int> &left_index,vector<int> &right_index,
                                    cv::Mat &mask,double dst_ratio,cv::Mat &E,
                                    double focal_lenght,double confidence,double reproject_err,Point2d pp,my_slam &obj)
{
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	vector<vector<DMatch>> matches;
    //my_slam::Ptr punt;
    cv::Mat _desc1=obj.descriptors[last_frame];
    cv::Mat _desc2=obj.descriptors[current_frame];
    vector<cv::KeyPoint> _features1=obj.features[last_frame];
    vector<cv::KeyPoint> _features2=obj.features[current_frame];
	matcher->knnMatch(_desc1, _desc2, matches, 2);
	for (unsigned int k = 0; k < matches.size(); k++)
	{
		if (matches[k][0].distance < dst_ratio * matches[k][1].distance)
		{ 
			corresponding_left.push_back(_features1[matches[k][0].queryIdx].pt);
			corresponding_right.push_back(_features2[matches[k][0].trainIdx].pt);
			left_index.push_back(matches[k][0].queryIdx);
			right_index.push_back(matches[k][0].trainIdx);
		}
	}
    E=findEssentialMat(corresponding_left,corresponding_right,focal_lenght,pp,RANSAC,confidence,reproject_err,mask);
}

void estimate_motion_and_calculate_3d_points(int last_frame,int current_frame,
                                            vector<Point2f> points_left,vector<Point2f> points_right,
                                            vector<int> index_left,vector<int> index_right,
                                            cv::Mat &intrinsic,cv::Mat &E,double focal_lenght,Point2d pp,
                                            double scale,cv::Mat &mask,vector<Point3d> &pts3d,my_slam &obj)
{
    //my_slam::Ptr punt;
    vector<Point2d> triangulation_points_left,triangulation_points_right;
    vector<int> inliers_recover_left,inliers_recover_right;
    Mat R,t;
    recoverPose(E,points_left,points_right,R,t,focal_lenght,pp,mask);
    for(int i=0;i<mask.rows;i++)
    {
        if(mask.at<unsigned char>(i))
        {
            triangulation_points_left.push_back(Point2d((double)points_left[i].x,(double)points_left[i].y));
            triangulation_points_right.push_back(Point2d((double)points_right[i].x,(double)points_right[i].y));
            inliers_recover_left.push_back(index_left[i]);
            inliers_recover_right.push_back(index_right[i]);
        }
    }
    //display for debug
    displayMatches(obj.imgs[last_frame],triangulation_points_left,obj.imgs[current_frame],triangulation_points_right);
    //do tracking
    cv::Mat last_pose=obj.cam_poses[last_frame];
    cv::Mat curr_rel_motion=generate_4x4_transformation(R,t);
    cv::Mat new_camera_pose=last_pose*curr_rel_motion.inv();
    relative_triangulation(triangulation_points_left,triangulation_points_right,intrinsic,R,t,pts3d);
    //map update && store camera_pose
    if(last_frame==0)
    {
        for(unsigned int i=0;i<pts3d.size();i++)
        {
            add_new_3d_point(last_frame,current_frame,pts3d[i],inliers_recover_left[i],inliers_recover_right[i],obj);
        }
        obj.cam_poses[current_frame]=new_camera_pose;
    }
    else
    {
        vector<int> identifiers,used_features;
        search_existent_points(obj.ident,inliers_recover_left,used_features,identifiers,obj);
        vector<Point3d> existent_3d,corresponding_3d;
        for(unsigned int i=0;i<used_features.size();i++)
        {
            if(used_features[i]==1)
            {
                existent_3d.push_back(obj.triangulated_points[identifiers[i]]);
                corresponding_3d.push_back(pts3d[i]);
            }    
        }
        update_scale(existent_3d,corresponding_3d,scale);
        t*=scale;
        relative_triangulation(triangulation_points_left,triangulation_points_right,intrinsic,R,t,pts3d);
        vector<Point3d> pts3d_in_world;
        for(unsigned int i=0;i<pts3d.size();i++)
        {
            cv::Point3d aux_point=change_points_to_other_ref_system(pts3d[i],last_pose);
            pts3d_in_world.push_back(aux_point);
        }
        curr_rel_motion=generate_4x4_transformation(R,t);
        new_camera_pose=last_pose*curr_rel_motion.inv();
        obj.cam_poses[current_frame]=new_camera_pose;
        for(unsigned int i=0;i<used_features.size();i++)
        {
            if(used_features[i]==0)
            {
                add_new_3d_point(last_frame,current_frame,pts3d_in_world[i],inliers_recover_left[i],inliers_recover_right[i],obj);
            }
            else
            {
                update_3d_point(obj,identifiers[i],inliers_recover_right[i],current_frame,pts3d_in_world[i]);
            }     
        }
    }
}
void read_poses_from_ground_truth(string path,int total_frames,vector<cv::Mat> &poses)
{
    ifstream myfile (path);
    if (myfile.is_open())
    {
        string line;
        for(int i=0;i<total_frames;i++)
        {
            Eigen::Quaterniond qa;
            Eigen::Vector3d trans;
            Eigen::Matrix4d eigen_pose=Eigen::Matrix4d::Identity();
            double aux;
            getline(myfile,line);
            std::istringstream in(line);  
            for(int j=0;j<8;j++)
            {     
                in >> aux;
                if(j==1) trans[0]=aux;
                if(j==2) trans[1]= aux;
                if(j==3) trans[2]= aux;
                if(j==4) qa.x()= aux;
                if(j==5) qa.y()= aux;
                if(j==6) qa.z()= aux;
                if(j==7) qa.w()= aux;
            } 
            eigen_pose.block<3,3>(0,0)=qa.normalized().toRotationMatrix();
            eigen_pose.block<3,1>(0,3)=trans;
            cv::Mat camera_pose=cv::Mat::eye(4,4,CV_64F);
            camera_pose.at<double>(0,0)=eigen_pose(0,0);
            camera_pose.at<double>(0,1)=eigen_pose(0,1);
            camera_pose.at<double>(0,2)=eigen_pose(0,2);
            camera_pose.at<double>(0,3)=eigen_pose(0,3);
            camera_pose.at<double>(1,0)=eigen_pose(1,0);
            camera_pose.at<double>(1,1)=eigen_pose(1,1);
            camera_pose.at<double>(1,2)=eigen_pose(1,2);
            camera_pose.at<double>(1,3)=eigen_pose(1,3);
            camera_pose.at<double>(2,0)=eigen_pose(2,0);
            camera_pose.at<double>(2,1)=eigen_pose(2,1);
            camera_pose.at<double>(2,2)=eigen_pose(2,2);
            camera_pose.at<double>(2,3)=eigen_pose(2,3);
            camera_pose.at<double>(3,0)=eigen_pose(3,0);
            camera_pose.at<double>(3,1)=eigen_pose(3,1);
            camera_pose.at<double>(3,2)=eigen_pose(3,2);
            camera_pose.at<double>(3,3)=eigen_pose(3,3);
            poses.push_back(camera_pose);
        }
    }
    myfile.close();
}      
void generate_ground_truth_for_comparation(vector<cv::Mat> &cam_poses)
{
    vector<cv::Mat> new_cam_poses;
    new_cam_poses.push_back(cv::Mat::eye(4,4,CV_64F));
    for(unsigned int i=1;i<cam_poses.size();i++)
    {
        cv::Mat prev=cam_poses[i-1];
        cv::Mat curr=cam_poses[i];
        cv::Mat rel_move=prev.inv()*curr;
        new_cam_poses.push_back(new_cam_poses[i-1]*rel_move);
    }
    std::ofstream file("./lectura_datos/generated_ground_truth.txt");
	for (unsigned int j = 0; j < new_cam_poses.size(); j++)
    {
        stringstream sss;
		string name;
		sss << j;
		name = sss.str();
		Eigen::Affine3f cam_pos;
        Eigen::Matrix4d eig_cam_pos=Eigen::Matrix4d::Identity();
        Eigen::Vector3d cam_translation;
        Eigen::Matrix3d cam_rotation;
        cv::Mat cv_cam_rot;
        cv::Mat cv_cam_tras;
        cv_cam_rot=new_cam_poses[j].rowRange(0,3).colRange(0,3);
        cv_cam_tras=new_cam_poses[j].rowRange(0,3).col(3);
        cv2eigen(cv_cam_rot,cam_rotation);
        cv2eigen(cv_cam_tras,cam_translation);
        eig_cam_pos.block<3,3>(0,0)=cam_rotation;
        eig_cam_pos.block<3,1>(0,3) = cam_translation;
        cam_pos=eig_cam_pos.cast<float>();
        Eigen::Quaternionf q(cam_pos.matrix().block<3,3>(0,0));

        file << j << " " << cam_pos(0,3) << " " <<  cam_pos(1,3) << " " << cam_pos(2,3) << " " << 
        q.x() << " " << q.y() << " " << q.z() << " " << q.w() << std::endl;
    }  
    file.close();
}
