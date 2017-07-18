import math
import numpy as np
from numpy import sin,pi,linspace
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad, cumtrapz, quad_explain
from scipy.signal import argrelextrema


g_SmoothingForParameterization_t = None
g_SmoothingForParameterization_s = None
g_SmoothingForDeltaCurvature = None


def getPointsAndFirstDerAtT(t, fx, fy):
	return fx([t])[0], fx.derivative(1)([t])[0], fy([t])[0], fy.derivative(1)([t])[0]
	
def lengthRateOfChangeFunc(t, fx, fy):
	x, dxdt, y, dydt = getPointsAndFirstDerAtT(t, fx, fy)
	val = math.sqrt(dxdt**2 + dydt**2)
	return val

def arcLengthAllTheWayToT(tList, fx_t, fy_t, noOfPoints=100, subDivide=1):
	all_x_vals = tList
	all_y_vals = []
	for i in range((len(all_x_vals)*subDivide)):
			x1 = float(i)/float(subDivide)
			all_y_vals.append(lengthRateOfChangeFunc(x1, fx_t, fy_t))
	
	#extract only the y values we care about
	next_all_y_vals = []
	for i in range(len(all_y_vals)/subDivide):
			next_all_y_vals.append(all_y_vals[i*subDivide])

	vals = cumtrapz(next_all_y_vals, all_x_vals, initial=0)
	return vals

def convertTListToArcLengthList(tList, fx_t, fy_t):
	return arcLengthAllTheWayToT(tList, fx_t, fy_t, noOfPoints=len(tList))

def getParameterizedFunctionFromPoints(tList, x_pts, y_pts, smoothing=None):
	fx_t = UnivariateSpline(tList, x_pts, k=3, s=smoothing)
	fy_t = UnivariateSpline(tList, y_pts, k=3, s=smoothing)
	return fx_t, fy_t
	
def reParameterizeFunctionFromPoints(tList, fx_t, fy_t, smoothing=None):
	#for each point (inputX[i], inputY[i]) the "arcLengthList" gives use the arc length from 0 to that point
	arcLengthList = convertTListToArcLengthList(tList, fx_t, fy_t)
	
	fx_s, fy_s = getParameterizedFunctionFromPoints(arcLengthList, fx_t(tList), fy_t(tList), smoothing=smoothing)
	return arcLengthList, fx_s, fy_s 

def getFirstAndSecondDerivForTPoints(arcLengthList, fx_s, fy_s):
	x = fx_s(arcLengthList)
	x_ = fx_s.derivative(1)(arcLengthList)
	x__ = fx_s.derivative(2)(arcLengthList)

	y = fy_s(arcLengthList)
	y_ = fy_s.derivative(1)(arcLengthList)
	y__ = fy_s.derivative(2)(arcLengthList)
	return x, x_, x__, y, y_, y__
	
#Note: curvature points won't be equidistant if the arcLengthList isn't
def calculateCurvature(arcLengthList, fx_s, fy_s, smoothing=None):
	x, x_, x__, y, y_, y__ = getFirstAndSecondDerivForTPoints(arcLengthList, fx_s, fy_s)
	curvature = abs(x_* y__ - y_* x__) / np.power(x_** 2 + y_** 2, 3 / 2)
	fCurvature = UnivariateSpline(arcLengthList, curvature, s=smoothing)
	return curvature
 
def parameterizeFunctionWRTArcLength(inputX, inputY):
		
	tList = np.arange(inputX.shape[0])
	fx_t, fy_t = getParameterizedFunctionFromPoints(tList, inputX, inputY, smoothing=g_SmoothingForParameterization_t)

	arcLengthList, fx_s, fy_s = reParameterizeFunctionFromPoints(tList, fx_t, fy_t, smoothing=g_SmoothingForParameterization_s)
		
	curvature = calculateCurvature(arcLengthList, fx_s, fy_s, smoothing=g_SmoothingForDeltaCurvature)

	return inputX, inputY, curvature

#Get the local maximums of curvature 
def getLocalMaximumsOfCurvature(pts, numberOfPixelsPerUnit=1):
	inputX, inputY = pts[:, 0], pts[:, 1]

	#set the scale
	inputX = np.multiply(inputX, 1./float(numberOfPixelsPerUnit))
	inputY = np.multiply(inputY, 1./float(numberOfPixelsPerUnit))
	
	#parameterize the function W.R.T. arc length
	xs, ys, curvature = parameterizeFunctionWRTArcLength(inputX, inputY)

	#get the local maximums of curvature
	localMaxima = argrelextrema(curvature, np.greater, order=2)

	localMaximaIndexes  = localMaxima[0]
	xsMaxima = xs[localMaximaIndexes]
	ysMaxima = ys[localMaximaIndexes]

	fin_pts = []
	for i in range(len(xsMaxima)):
			pt = (xsMaxima[i], ysMaxima[i])
			fin_pts.append(pt)

	return [xsMaxima], [ysMaxima]

