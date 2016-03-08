#include "spray.h"
#include <windows.h>
#include <stdlib.h>
#include <fstream>
#include <math.h>
#include <string.h>
#include <iostream>
using namespace std;


//read solid data, added by GY.
void cspray::readdata_solid(void)
{
	const int lineLength = 100;
	char str[lineLength];

	ifstream dataFile;
	if (mscene == SCENE_MELTANDBOIL_HIGHRES || mscene == SCENE_INTERACTION_HIGHRES)
		dataFile.open("./PointCloud/bunny_4X.asc");
	else
		dataFile.open("./PointCloud/bunny.asc");

	if (!dataFile)
	{
		cout << "文件路径有误，读取失败，按任意键退出。" << endl;
		getchar();
		exit(0);
	}
	cout << "读取数据成功" << endl;

	dataFile >> nInitSolPoint;
	nRealSolpoint = nInitSolPoint;
	cout << "文件包含" << nInitSolPoint << "个点数据" << endl;

	//读取数据点坐标
	dataFile.clear();
	dataFile.seekg(0, ios::beg);
	SolpointPos = new double*[nInitSolPoint];
	for (int i = 0; i < nInitSolPoint; i++)
	{
		SolpointPos[i] = new double[3];
	}

	for (int i = 0; !dataFile.eof(); i++)
	{
		dataFile.getline(str, lineLength);
		dataFile >> Pointnum;
		dataFile.get();
		for (int j = 0; j < 3; j++)
		{

			(dataFile) >> SolpointPos[i][j];
			dataFile.get();
		}
	}
	dataFile.close();
	return;
}
