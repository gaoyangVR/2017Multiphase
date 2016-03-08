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
		cout << "�ļ�·�����󣬶�ȡʧ�ܣ���������˳���" << endl;
		getchar();
		exit(0);
	}
	cout << "��ȡ���ݳɹ�" << endl;

	dataFile >> nInitSolPoint;
	nRealSolpoint = nInitSolPoint;
	cout << "�ļ�����" << nInitSolPoint << "��������" << endl;

	//��ȡ���ݵ�����
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
