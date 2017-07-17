import math
import numpy as np
from numpy import sin,pi,linspace
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad, cumtrapz, quad_explain
from scipy.signal import argrelextrema

g_SmoothingForParameterization_t = None
g_SmoothingForParameterization_s = None
g_SmoothingForDeltaCurvature = None
g_isMakeAllPointsEqidistant = False
g_cullPoints = False
g_maxNoOfPointsForCullingFunctoin = 40
g_SmoothingForPointsCulling = 0
g_numberOfPixelsPerUnit = 1

#debug
g_dividerForPts = 1

def breakUpFullLengthOfArcIntoXPoints(fullLength, noOfPoints, addZeroPoint=False):
	step = float(fullLength)/float(noOfPoints)
	ret = []
	tempVal = 0
	if addZeroPoint:
		ret.append(0)

	for i in range(noOfPoints):
		tempVal += step
		ret.append(tempVal)
		
	return ret

#express the function in less points by parameterizing WRT some variable (t) 
#and then interpolating
def getSimplePts(pts, maxNoOfPoints=g_maxNoOfPointsForCullingFunctoin):
	org_x, org_y = pts[:, 0], pts[:, 1]
	tList = np.arange(org_x.shape[0])
	fx_t = UnivariateSpline(tList, org_x, k=3, s=g_SmoothingForPointsCulling)
	fy_t = UnivariateSpline(tList, org_y, k=3, s=g_SmoothingForPointsCulling)
	newTList = breakUpFullLengthOfArcIntoXPoints(tList[-1], maxNoOfPoints, addZeroPoint=True)
	xt = fx_t(newTList)
	yt = fy_t(newTList)
	return xt, yt, newTList

def getPointsAndFirstDerAtT(t, fx, fy):
	return fx([t])[0], fx.derivative(1)([t])[0], fy([t])[0], fy.derivative(1)([t])[0]
	
def lengthRateOfChangeFunc(t, fx, fy):
	x, dxdt, y, dydt = getPointsAndFirstDerAtT(t, fx, fy)
	val = math.sqrt(dxdt**2 + dydt**2)
	return val

def arcLengthAllTheWayToT(tList, fx_t, fy_t, noOfPoints=100, subDivide=g_dividerForPts):
	all_x_vals = tList
	all_y_vals = []
	for i in range((len(all_x_vals)*subDivide)):#FIXME: this won't work for odd values of tList, it makes the presumption that tList is just incrementing ints
			x1 = float(i)/float(subDivide)
			all_y_vals.append(lengthRateOfChangeFunc(x1, fx_t, fy_t))
	
	#extract only the y values we care about
	next_all_y_vals = []
	for i in range(len(all_y_vals)/subDivide):
			next_all_y_vals.append(all_y_vals[i*subDivide])

	all_y_vals = next_all_y_vals

	vals = cumtrapz(all_y_vals, all_x_vals, initial=0)
	return vals

def convertTListToArcLengthList(tList, fx_t, fy_t):
	return arcLengthAllTheWayToT(tList, fx_t, fy_t, noOfPoints=len(tList))

def convertTListToArcLengthList_debug_new(tList, fx_t, fy_t):
	return convertTListToArcLengthList(tList, fx_t, fy_t)

def getPointsAndFirstDerAtT_old(t, fx, fy):
	return fx([t])[0], fx.derivative(1)([t])[0], fy([t])[0], fy.derivative(1)([t])[0]
	
def lengthRateOfChangeFunc_old(t, fx, fy):
	x, dxdt, y, dydt = getPointsAndFirstDerAtT_old(t, fx, fy)
	val = math.sqrt(dxdt**2 + dydt**2)
	return val

def arcLengthAtParamT(t, fx_t, fy_t):
	val = quad(lengthRateOfChangeFunc_old, 0, t, args=(fx_t, fy_t))
	return val[0]

def getParameterizedFunctionFromPoints(tList, x_pts, y_pts, smoothing=None):
	fx_t = UnivariateSpline(tList, x_pts, k=3, s=smoothing)
	fy_t = UnivariateSpline(tList, y_pts, k=3, s=smoothing)
	return fx_t, fy_t
	
def reParameterizeFunctionFromPoints(tList, fx_t, fy_t, smoothing=None):
	#for each point (org_x[i], org_y[i]) the "arcLengthList" gives use the arc length from 0 to that point
	arcLengthList = convertTListToArcLengthList_debug_new(tList, fx_t, fy_t)
	
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

def newArcLengthList(oldArcLengthList, fx_s, fy_s, isMakeAllPointsEqidistant=g_isMakeAllPointsEqidistant):
	if isMakeAllPointsEqidistant:
		return getEqidistantPointsAlongFunction(oldArcLengthList, fx_s, fy_s)
	return oldArcLengthList
	
#Remember: curvature points will be the input points (and so won't be equidistant if the arcLengthList isn't)
def getCurvatureForPoints(arcLengthList, fx_s, fy_s, smoothing=None):
	x, x_, x__, y, y_, y__ = getFirstAndSecondDerivForTPoints(arcLengthList, fx_s, fy_s)
	curvature = abs(x_* y__ - y_* x__) / np.power(x_** 2 + y_** 2, 3 / 2)
	fCurvature = UnivariateSpline(arcLengthList, curvature, s=smoothing)
	dxcurvature = fCurvature.derivative(1)(arcLengthList)
	dx2curvature = fCurvature.derivative(2)(arcLengthList)
	return curvature, dxcurvature, dx2curvature
	
def parameterizeFunctionWRTArcLength(pts):
	org_x, org_y = pts[:, 0], pts[:, 1]
	return _parameterizeFunctionWRTArcLength(org_x, org_y)

def _parameterizeFunctionWRTArcLength(org_x, org_y):
		
	tList = np.arange(org_x.shape[0])
	fx_t, fy_t = getParameterizedFunctionFromPoints(tList, org_x, org_y, smoothing=g_SmoothingForParameterization_t)

	arcLengthList, fx_s, fy_s = reParameterizeFunctionFromPoints(tList, fx_t, fy_t, smoothing=g_SmoothingForParameterization_s)
	
	arcLengthList = newArcLengthList(arcLengthList, fx_s, fy_s)
	x, x_, x__, y, y_, y__ = getFirstAndSecondDerivForTPoints(arcLengthList, fx_s, fy_s)
	
	curvature, dxcurvature, dx2curvature = getCurvatureForPoints(arcLengthList, fx_s, fy_s, smoothing=g_SmoothingForDeltaCurvature)

	return org_x, org_y, x_, y_, x__, y__, arcLengthList, curvature, dxcurvature, dx2curvature, arcLengthList[-1], fx_s, fy_s

def genImagesWithDisplayFix(pts, numberOfPixelsPerUnit=g_numberOfPixelsPerUnit):
	org_x, org_y = pts[:, 0], pts[:, 1]

	if g_cullPoints:
		org_x, org_y, junk = getSimplePts(pts)


	org_x = np.multiply(org_x, 1./float(numberOfPixelsPerUnit))
	org_y = np.multiply(org_y, 1./float(numberOfPixelsPerUnit))
	
	xs, ys, dxdt, dydt, d2xdt, d2ydt, s, curvature, dxcurvature, dx2curvature, fullLength_s, fx_s, fy_s =_parameterizeFunctionWRTArcLength(org_x, org_y)

	finalVals = argrelextrema(curvature, np.greater, order=2)

	temp = []
	temp2 = []
	temp3 = []
	temp4 = []

	I__  = finalVals[0]
	temp = curvature[I__]
	temp2 = xs[I__]
	temp3 = ys[I__]
	temp4 = s[I__]

	fin_pts = []
	for i in range(len(temp2)):
			pt = (temp2[i], temp3[i])
			fin_pts.append(pt)


	return [temp2], [temp3]

def genImages(pts):
	#simplify the points
	new_org_x, new_org_y, new_tList = getSimplePts(pts, maxNoOfPoints=100)

	org_x, org_y = new_org_x, new_org_y#pts[:, 0], pts[:, 1]

	xs, ys, dxdt, dydt, d2xdt, d2ydt, s, curvature, dxcurvature, dx2curvature, fullLength_s = _parameterizeFunctionWRTArcLength(org_x, org_y)

	dx2curvature = abs(dx2curvature)
	maxm = argrelextrema(dx2curvature, np.greater)  # (array([1, 3, 6]),)

	coordsx = []
	coordsy = []
	for val in maxm:
		coordsx.append(xs[val])
		coordsy.append(ys[val])

	return coordsx, coordsy



	
