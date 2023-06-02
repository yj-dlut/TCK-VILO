#include "vilo/file_reader.h"
using namespace std;


namespace vilo {

FileReader::FileReader(std::string image_folder, cv::Size new_size, std::string time_folder)
{
    getDir(image_folder, m_image_files);
    printf("FileReader: got %d files in %s!\n", (int)m_image_files.size(), image_folder.c_str());

    m_img_new_size = new_size;

    m_stamp_valid = false;
    if(time_folder != "None")
    {   
        std::ifstream tr;
        tr.open(time_folder.c_str());
        while(!tr.eof() && tr.good())
        {
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            char stamp[100];
            float x,y,z,a,b,c,d;
            float exposure = 0;
  

            if(8 == sscanf(buf, "%s %f %f %f %f %f %f %f", stamp, &x, &y, &z, &a, &b, &c, &d))
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(3 == sscanf(buf, "%d %s %f", &id, stamp, &exposure)) 
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(2 == sscanf(buf, "%d %s", &id, stamp))  
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }

        }
        tr.close();

        assert(m_times.size() == m_image_files.size());
        m_stamp_valid = true;
    }
}
FileReader::FileReader(std::string image_folder, std::string time_folder)
{
    getDir(image_folder, m_image_files);
    printf("FileReader: got %d files in %s!\n", (int)m_image_files.size(), image_folder.c_str());

    m_stamp_valid = false;
    if(time_folder != "None")
    {   
        std::ifstream tr;
        tr.open(time_folder.c_str());
        while(!tr.eof() && tr.good())
        {
            char buf[1000];
            tr.getline(buf, 1000);

            int id;
            char stamp[100];
            float x,y,z,a,b,c,d;
            float exposure = 0;
  

            if(8 == sscanf(buf, "%s %f %f %f %f %f %f %f", stamp, &x, &y, &z, &a, &b, &c, &d))
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(3 == sscanf(buf, "%d %s %f", &id, stamp, &exposure)) 
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }
            else if(2 == sscanf(buf, "%d %s", &id, stamp))  
            {
                std::string time_stamp = stamp;
                m_times.push_back(time_stamp);
            }

        }
        tr.close();

        assert(m_times.size() == m_image_files.size());
        m_stamp_valid = true;
    }
}

FileReader::FileReader()
{
    m_stamp_valid = false;
}

int FileReader::readEurocFilesTimes(std::string file_folder, 
                            std::vector<std::string>& file_name_vec,
                            std::vector<double>& time_vec)
{
    int N = getDir(file_folder, m_image_files);
    printf("FileReader: got %d files in %s!\n", N, file_folder.c_str());

    time_vec.resize(N);
    file_name_vec.resize(N);
    for(int i=0;i<N;i++)
    {
        std::stringstream ss;
        ss << m_image_files[i];

        std::string str;
        ss>>str;
        file_name_vec[i] = str;

        int sid = str.find_last_of('/');
        int eid = str.find_last_of('.');
        std::string time = str.substr(sid+1, eid-sid-1);
        std::stringstream ss1;
        ss1<<time;
        double t;
        ss1 >> t;
        time_vec[i] = t/1e9;
    }
    m_stamp_valid = true;
    return N;
}

int FileReader::readEurocImu(std::string imu_folder, 
                     std::vector<Eigen::Vector3d>& imu_acc_vec,
                     std::vector<Eigen::Vector3d>& imu_gyro_vec,
                     std::vector<double>& time_vec)
{
    std::ifstream in;
    in.open(imu_folder.c_str());
    time_vec.reserve(200000);
    imu_acc_vec.reserve(200000);
    imu_gyro_vec.reserve(200000);

    while(!in.eof())
    {
        std::string s;
        getline(in,s);
        if (s[0] == '#')
            continue;

        if(!s.empty())
        {
            std::string item;
            size_t pos = 0;
            double data[7];
            int count = 0;
            while ((pos = s.find(',')) != std::string::npos) {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[6] = stod(item);

            time_vec.push_back(data[0]/1e9);
            imu_acc_vec.push_back(Eigen::Vector3d(data[4],data[5],data[6]));
            imu_gyro_vec.push_back(Eigen::Vector3d(data[1],data[2],data[3]));
        }
    }  
    return (int)time_vec.size();
}

int FileReader::readFileNamesTimes(std::string file_folder, 
                                    std::vector<std::string>& file_name_vec,
                                    std::vector<double>& time_vec,
                                    int type)
{
    int N = ( type==1 ? getDir(file_folder, m_image_files) : getDir(file_folder, m_lidar_files) );
    printf("FileReader: got %d files in %s!\n", N, file_folder.c_str());

    time_vec.resize(N);
    file_name_vec.resize(N);
    for(int i=0;i<N;i++)
    {
        std::stringstream ss;
        if(type==1)
            ss << m_image_files[i];
        else 
            ss << m_lidar_files[i];

        std::string str;
        ss>>str;
        file_name_vec[i] = str;
     
        int id = str.find_last_of('/');
        string sub_str = str.substr(id+1, str.length()-1); 
        int id1 = sub_str.find_first_of('.');
        int eid = sub_str.find_last_of('.');
        string str_sec = sub_str.substr(0, id1);
        string str_nsec = sub_str.substr(id1+1, eid);
        std::stringstream ss1; ss1<<str_sec;
        double sec; ss1>>sec;
        std::stringstream ss2; ss2<<str_nsec;
        double nsec; ss2>>nsec;
        time_vec[i] = sec + 1e-6 * nsec;
    }
    m_stamp_valid = true;
    return N;
}

int FileReader::readFileNames(std::string file_folder, 
                                    std::vector<std::string>& file_name_vec,
                                    int type)
{
    int N = ( type==1 ? getDir(file_folder, m_image_files) : getDir(file_folder, m_lidar_files) );
    printf("FileReader: got %d files in %s!\n", N, file_folder.c_str());

    file_name_vec.resize(N);
    for(int i=0;i<N;i++)
    {
        std::stringstream ss;
        if(type==1)
            ss << m_image_files[i];
        else 
            ss << m_lidar_files[i];

        std::string str;
        ss>>str;
        file_name_vec[i] = str;
    }
    m_stamp_valid = false;
    return N;
}

int FileReader::readFileTimes(std::string file_folder, std::vector<double>& time_vec)
{
    std::ifstream in;
    in.open(file_folder.c_str());
    string str;
    while(getline(in, str))
    {
        time_vec.push_back(stof(str.c_str()));
    }
    
    m_stamp_valid = true;
    return time_vec.size();
}

int FileReader::readImu(std::string imu_folder, 
                         std::vector<Eigen::Vector3d>& imu_acc_vec,
                         std::vector<Eigen::Vector3d>& imu_gyro_vec,
                         std::vector<double>& time_vec)
{
    std::ifstream in;
    in.open(imu_folder.c_str());
    time_vec.reserve(200000);
    imu_acc_vec.reserve(200000);
    imu_gyro_vec.reserve(200000);

    while(!in.eof())
    {
        std::string s;
        getline(in,s);
        if (s[0] == '#')
            continue;    
        if(!s.empty())
        {
            
            std::string item;
            size_t pos = 0;
            double data[9];
            int count = 0;
            while ((pos = s.find(',')) != std::string::npos) 
            {
                item = s.substr(0, pos);
                data[count++] = stod(item);
                s.erase(0, pos + 1);
            }
            item = s.substr(0, pos);
            data[8] = stod(item);

            time_vec.push_back(data[1] + data[2]*10e-10);
            imu_gyro_vec.push_back(Eigen::Vector3d(data[3],data[4],data[5]));
            imu_acc_vec.push_back(Eigen::Vector3d(data[6],data[7],data[8]));
        }
    }
    return (int)time_vec.size();
}

int FileReader::readImuTXT(std::string imu_folder, 
                         std::vector<Eigen::Vector3d>& imu_acc_vec,
                         std::vector<Eigen::Vector3d>& imu_gyro_vec,
                         std::vector<double>& time_vec)
{
    std::ifstream in;
    in.open(imu_folder.c_str());
    time_vec.reserve(200000);
    imu_acc_vec.reserve(200000);
    imu_gyro_vec.reserve(200000);

    double t, ax, ay, az, rx, ry, rz;
    while(in>>t>>ax>>ay>>az>>rx>>ry>>rz)
    {
        time_vec.push_back(t);
        imu_gyro_vec.push_back(Eigen::Vector3d(rx, ry, rz));
        imu_acc_vec.push_back(Eigen::Vector3d(ax, ay, az));
    }
    
    return (int)time_vec.size();
}

cv::Mat FileReader::readImage(int image_index , cv::Size new_size)
{
    m_img_new_size = new_size;
    cv::Mat image = cv::imread(m_image_files.at(image_index), CV_LOAD_IMAGE_GRAYSCALE);
        
    if(!image.data)
    {
        std::cout << "ERROR READING IMAGE " << m_image_files.at(image_index) << std::endl;
        return cv::Mat();
    }
    
    cv::resize(image, image, m_img_new_size);
    
    return image;
}

cv::Mat FileReader::readImage(int image_index)
{
    cv::Mat image = cv::imread(m_image_files.at(image_index), CV_LOAD_IMAGE_GRAYSCALE);
        
    if(!image.data)
    {
        std::cout << "ERROR READING IMAGE " << m_image_files.at(image_index) << std::endl;
        return cv::Mat();
    }

    cv::resize(image, image, m_img_new_size);
    
    return image;
}

std::string FileReader::readStamp(int image_index)
{
    return m_times[image_index];
}

int FileReader::getDir(std::string dir, std::vector<std::string> &files)
{
    DIR *dp;
    struct dirent *dirp;
    if((dp = opendir(dir.c_str())) == NULL)
    {
        return -1;
    }

    while ((dirp = readdir(dp)) != NULL)
    {
        std::string name = std::string(dirp->d_name);

        if(name != "." && name != "..")
        {
            if(strstr(name.c_str(),".png")!=NULL 
                || strstr(name.c_str(),".jpg")!=NULL 
                   || strstr(name.c_str(),".bmp")!=NULL
                      || strstr(name.c_str(),".pcd")!=NULL
                         || strstr(name.c_str(),".txt")!=NULL
                            || strstr(name.c_str(),".bin")!=NULL)
                files.push_back(name);
        }
    }

    closedir(dp);
    std::sort(files.begin(), files.end());

    if(dir.at(dir.length() - 1) != '/')
        dir = dir+"/";

    for(unsigned int i = 0; i < files.size(); i++)
    {
        if(files[i].at(0) != '/')
            files[i] = dir + files[i];
    }

    return (int)files.size();
}

void FileReader::readPointsFromTXT(int id, std::vector<double>& pts_ts_vec)
{
    struct timeval st,et; 
    gettimeofday(&st,NULL);

	std::ifstream in(m_lidar_files[id],  ios::binary);
	if (in.bad())
    {
		return;
	}

    pcl::PointCloud<pcl::PointXYZI>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZI>);
    points2->points.reserve(65536+1);
    pts_ts_vec.reserve(65536+1);
    pcl::PointXYZI p;
    
    double x, y, z, intensity, ring, delay;
    while(in>>x>>y>>z>>intensity>>ring>>delay)
    {
        p.x = x;
        p.y = y;
        p.z = z;
        p.intensity = intensity;
        points2->points.push_back(p);
        pts_ts_vec.push_back(delay*1e-9);
    }

    cloud_ = points2;
    
    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);  
}

void FileReader::readPointsFromPCD(int id, std::vector<double>& pts_ts_vec)
{
    struct timeval st,et; 
    float time_use = 0;

	std::ifstream in(m_lidar_files[id],  ios::binary);
	if (in.bad())
    {
		return;
	}
 
	char s[11][1024]; 
	int N = 0; 
	std::string data_columns_type; 
	std::string data_type; 

	for (int i = 0; i < 11; ++i)
    {
		in.getline(s[i], 1024);
        
		if (i == 2)
        {
			std::string s2 = s[2];
			size_t pos = s2.find("FIELDS");
			size_t size = s2.size();
			data_columns_type = s2.substr(pos + 7, size);
		}
 
		if (i == 9)
        {
			std::string s9 = s[9], Points_Str;
			size_t pos = s9.find("POINTS");
			size_t size = s9.size();
			Points_Str = s9.substr(pos + 7, size);
			N = atoi(Points_Str.c_str());
		}
 
		if (i == 10)
        {
			std::string s10 = s[10], DATA_SIZE;
			size_t pos = s10.find("DATA");
			size_t size = s10.size();
			data_type = s10.substr(pos + 5, size);
		}
	}

    pcl::PointCloud<pcl::PointXYZI>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZI>);
    points2->points.reserve(N+1);
    pts_ts_vec.reserve(N+1);
	pcl::PointXYZI p;
	if ((data_columns_type == "x y z intensity t reflectivity ring noise range") && (data_type == "binary"))
    {
        double x, y, z, intensity, t, reflectivity, ring, noise, range; 
        
		while (!in.eof())
        {
			in >> p.x >> p.y >> p.z >> intensity >> t >> reflectivity >> p.intensity >> noise >> range;
            pts_ts_vec.push_back(t);
            
            if (in.peek() == EOF)
            {
				break;
			}
			points2->points.push_back(p);
		}
	}
    else
    {
		std::cout << "data_type = ascii, read failed!" << std::endl;
    }

    cloud_ = points2;
}

void FileReader::readPointsFromBIN(int laser_index)
{
    struct timeval st,et; 
    int npts = 0;   
    float time_use = 0;
    std::string in_file = m_lidar_files[laser_index];

    npts = 0;
    int32_t num = 1000000;
    float *data = new float[num];
    
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pr = data+3;

    FILE *stream;
    stream = fopen (in_file.c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/4;
    pcl::PointCloud<pcl::PointXYZI>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZI>);
    
    for (int32_t i=0; i<num; i++)
    {
        px+=4; py+=4; pz+=4; pr+=4;
        pcl::PointXYZI pt;
        pt.x = *px;
        pt.y = *py;
        pt.z = *pz;
        pt.intensity = *pr;
        points2->points.push_back(pt);
        npts++;

    }
    
    fclose(stream);
    delete []data;
    
    cloud_ = points2;
 }

void FileReader::readPointsFromBIN(int id, std::vector<double>& pts_ts_vec)
{
    struct timeval st,et; 
    gettimeofday(&st,NULL);

    int npts = 0;   
    long num = 1000000;
    float *data = new float[num];
    
    float *px = data+0;
    float *py = data+1;
    float *pz = data+2;
    float *pi = data+3;
    float *pr = data+4;
    float *pd = data+5;

    FILE *stream;
    std::string in_file = m_lidar_files[id];
    stream = fopen (in_file.c_str(),"rb");
    num = fread(data,sizeof(float),num,stream)/6;

    pcl::PointCloud<pcl::PointXYZI>::Ptr points2 (new pcl::PointCloud<pcl::PointXYZI>);
    points2->points.reserve(65536+1);
    pts_ts_vec.reserve(65536+1);
  
    for (long i=0; i<num-1; i++)
    {
        px+=6; py+=6; pz+=6; pi+=6; pr+=6; pd+=6;
        
        pcl::PointXYZI pt;
        pt.x = *px;
        pt.y = *py;
        pt.z = *pz;
        pt.intensity = *pr;
        points2->points.push_back(pt);
        pts_ts_vec.push_back((*pd)*1e-9);
        npts++;
    }
    
    fclose(stream);
    delete []data;
    
    cloud_ = points2;

    gettimeofday(&et,NULL);
    float time_use = (et.tv_sec-st.tv_sec)*1000000+(et.tv_usec-st.tv_usec);  
}

void FileReader::readFromTXTSaveToBin(int i)
{
    std::ifstream in(m_lidar_files[i],  ios::binary);
	if (in.bad())
    {
		return;
	}

    std::string txt_str = m_lidar_files[i];
    int id = txt_str.find_last_of('/');
    std::string sub_str = txt_str.substr(id+1, txt_str.length()-1);
    std::string bin_str = "/media/jun/新加卷/data/The_Newer_College/lidar_bin/" 
                           + sub_str.substr(0,sub_str.length()-3) + "bin";

    FILE *fid;
    fid = fopen(bin_str.c_str(),"wb");
    if(fid == NULL)
    {
        return;
    }

    long npt = 0;
    float x, y, z, intensity, ring, delay;
    std::vector<float> pts_vec;
    while(in>>x>>y>>z>>intensity>>ring>>delay)
    {  
        pts_vec.push_back(x); pts_vec.push_back(y); 
        pts_vec.push_back(z); pts_vec.push_back(intensity); 
        pts_vec.push_back(ring); pts_vec.push_back(delay); 
    }

    long n_data = pts_vec.size();
    float *data = new float[n_data];
    for(long i=0;i<n_data;i++)
        data[i] = pts_vec[i];
    fwrite(data, sizeof(float), n_data, fid);
}

}
