import time

import math
from scipy.interpolate import UnivariateSpline, interp1d
from scipy.integrate import quad, cumtrapz, quad_explain
import numpy as np
import pylab
from numpy import sin,pi,linspace
from pylab import plot,show,subplot, axhline, axis, axes
from scipy.signal import argrelextrema

g_name = 'REPLACE_ME'
g_enable_plotting = True
g_SmoothingForParameterization_t = None
g_SmoothingForParameterization_s = None
g_SmoothingForDeltaCurvature = None
g_isMakeAllPointsEqidistant = False
g_cullPoints = False
g_maxNoOfPointsForCullingFunctoin = 40
g_SmoothingForPointsCulling = 0
g_numberOfPixelsPerUnit = 1

#debug
g_plotVelocity = True
g_dividerForPts = 1

def PointsInCircum(r,n=100):
    return [(math.cos(2*pi/n*x)*r,math.sin(2*pi/n*x)*r) for x in xrange(0,n+1)]
	
def thatCurve(a=9, b=6, noOfPoints=200):
	delta = 0#pi/2
	div = max(a,b)
	end = -float(pi*(float(b)/float(a)))
	offset = -float(pi*(float(b)/float(a)))/4
	t = linspace(offset,end+offset,noOfPoints)


	x = sin(a * t + delta)
	y = sin(b * t)

	#subplot(223)
	#plot(x,y, 'b', color="blue")
	ret = []
	for i in range(len(x)):
		ret.append( (x[i], y[i]) )
	return ret

def plotVelocity(t_pts, v_pts):
	plot(t_pts, v_pts)
	show()
#####################################
####### debug prints #######
#####################################
def printTheDerivativesError(x_, y_):
	print "x_**2 + y_**2"
	vals = x_**2 + y_**2
	print vals
	print "total error: " + str(np.sum(abs(vals-1)))
	
def randomPrint1(arcLengthList, org_x, org_y, fx_s, fy_s):
	print 'now the functions'
	print arcLengthList
	print org_x
	print org_y
	print fx_s(arcLengthList)
	print fy_s(arcLengthList)	

#####################################
####### make less points
#####################################
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
#####################################
####### \make less points
#####################################

#####################################
####### main code ########	
#####################################

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
 	#print "starting the integral"
	time1 = time.time()
	arcLengthList = arcLengthAllTheWayToT(tList, fx_t, fy_t, noOfPoints=len(tList))
	time2 = time.time()
	print 'integral took %0.3f seconds' % ((time2-time1))
	return arcLengthList

def plotTheIntegratorStuff(ptsSoFar, fx_t, fy_t, all_y_vals, idx, expandedTList_ptsSoFar, expandedTList_all_y_vals):

	subplot(211)
	plot(ptsSoFar, all_y_vals, 'b', color='r')
	plot(expandedTList_ptsSoFar, expandedTList_all_y_vals, 'b',  color='b')
	#subplot(422)
	#plot(ptsSoFar, fx_t(ptsSoFar))
	#subplot(423)
	#plot(ptsSoFar, fy_t(ptsSoFar))
	subplot(212)
	plot(fx_t(ptsSoFar), fy_t(ptsSoFar))
	pylab.savefig('output_debug/'+g_name+'_foo'+str(idx)+'.png')
	pylab.clf()


def plotThisPointForTheArcLengthStuff(ptsSoFar, fx_t, fy_t, idx, dividerForPts=g_dividerForPts):
	all_y_vals = []
	for x1 in ptsSoFar:
		all_y_vals.append(lengthRateOfChangeFunc(x1, fx_t, fy_t))

	expandedYVals = []
	expandedTVals = []
	for i in range((len(ptsSoFar)*dividerForPts)):
		expandedTVals.append(float(i)/float(dividerForPts))
		expandedYVals.append(lengthRateOfChangeFunc(float(i)/float(dividerForPts), fx_t, fy_t))

	#remember to plot the rate of change function
	plotTheIntegratorStuff(ptsSoFar, fx_t, fy_t, all_y_vals, idx, expandedTVals, expandedYVals)


def convertTListToArcLengthList_debug_new(tList, fx_t, fy_t):
#	ptsSoFar = []
#
#	all_y_vals = []
#	for x1 in tList:
#		all_y_vals.append(lengthRateOfChangeFunc(x1, fx_t, fy_t))
#
#	expandedYVals = []
#	expandedTVals = []
#	dividerForPts = g_dividerForPts
#	for i in range((len(tList)*dividerForPts)):
#		expandedTVals.append(float(i)/float(dividerForPts))
#		expandedYVals.append(lengthRateOfChangeFunc(float(i)/float(dividerForPts), fx_t, fy_t))
#
#	#plot(expandedTVals, expandedYVals, 'b', color='b')
#	#plot(tList, all_y_vals, 'b', color='r')
#	#show()
#
#	for i in range(len(tList)):
#		ptsSoFar.append(tList[i])
#		if i == 0:
#			continue
#
#		plotThisPointForTheArcLengthStuff(ptsSoFar, fx_t, fy_t, i)


	return convertTListToArcLengthList(tList, fx_t, fy_t)



################################################
##############OLD

def getPointsAndFirstDerAtT_old(t, fx, fy):
	return fx([t])[0], fx.derivative(1)([t])[0], fy([t])[0], fy.derivative(1)([t])[0]
	
def lengthRateOfChangeFunc_old(t, fx, fy):
	x, dxdt, y, dydt = getPointsAndFirstDerAtT_old(t, fx, fy)
	val = math.sqrt(dxdt**2 + dydt**2)
	return val

def arcLengthAtParamT(t, fx_t, fy_t):
	val = quad(lengthRateOfChangeFunc_old, 0, t, args=(fx_t, fy_t))
	return val[0]

def TtoS_old(tList, fx, fy):
	ret = []
	for val in tList:
		ret.append(arcLengthAtParamT_old(val, fx, fy))

	return np.array(ret)
	

def _convertTListToArcLengthList_old(tList, fx, fy):
	return TtoS_old(tList, fx, fy)
	
def convertTListToArcLengthList_old(tList, fx_t, fy_t):
# 	print "starting the integral"
	time1 = time.time()
	arcLengthList = _convertTListToArcLengthList_old(tList, fx_t, fy_t)
	time2 = time.time()
	print 'integral took %0.3f seconds' % ((time2-time1))
	return arcLengthList

################################################
##############OLD END

def arcLengthAtParamT_old(t, fx_t, fy_t):
	val = quad(lengthRateOfChangeFunc, 0, t, args=(fx_t, fy_t))
	return val[0]

def getParameterizedFunctionFromPoints(tList, x_pts, y_pts, smoothing=None):
	fx_t = UnivariateSpline(tList, x_pts, k=3, s=smoothing)
	fy_t = UnivariateSpline(tList, y_pts, k=3, s=smoothing)
	return fx_t, fy_t
	
#if reparameterizing with respect to arc length then [s1, s2, ...] = t_to_s_function([t1, t2, ...])
def reParameterizeFunctionFromPoints(t_to_s_function, tList, fx_t, fy_t, smoothing=None):
	#for each point (org_x[i], org_y[i]) the "arcLengthList" gives use the arc length from 0 to that point
	#arcLengthList = convertTListToArcLengthList_old(tList, fx_t, fy_t)
	
	arcLengthList = convertTListToArcLengthList_debug_new(tList, fx_t, fy_t)
	
#	print '############'
#	print arcLengthList
#	print arcLengthList2
#	print len(arcLengthList)
#	print len(arcLengthList2)
#	print '############'
	
#	sys.exit()
	#CAREFUL!!! we might use fx_t(tList) here, or x_org 
	#but only if that's what the arcLengthListIsDefinedAS ??? it must be, how else would we make the spline????
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

def getEqidistantPointsArcLengthList(oldArcLengthList, fx_s, fy_s):
	#TODO: implement this if needed
	pass
	
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

	arcLengthList, fx_s, fy_s = reParameterizeFunctionFromPoints(convertTListToArcLengthList_old, tList, fx_t, fy_t, smoothing=g_SmoothingForParameterization_s)
	
	arcLengthList = newArcLengthList(arcLengthList, fx_s, fy_s)
	x, x_, x__, y, y_, y__ = getFirstAndSecondDerivForTPoints(arcLengthList, fx_s, fy_s)
	
	curvature, dxcurvature, dx2curvature = getCurvatureForPoints(arcLengthList, fx_s, fy_s, smoothing=g_SmoothingForDeltaCurvature)

	return org_x, org_y, x_, y_, x__, y__, arcLengthList, curvature, dxcurvature, dx2curvature, arcLengthList[-1], fx_s, fy_s

def genImages2(retX, retY):
	return genImages

def genImagesWithDisplayFix(pts, numberOfPixelsPerUnit=g_numberOfPixelsPerUnit):
	org_x, org_y = pts[:, 0], pts[:, 1]

	if g_cullPoints:
		org_x, org_y, junk = getSimplePts(pts)


	org_x = np.multiply(org_x, 1./float(numberOfPixelsPerUnit))
	org_y = np.multiply(org_y, 1./float(numberOfPixelsPerUnit))
	
	#print 'Vals of org_x and org_y after mult'
	#print org_x
	#print org_y
	
	xs, ys, dxdt, dydt, d2xdt, d2ydt, s, curvature, dxcurvature, dx2curvature, fullLength_s, fx_s, fy_s =_parameterizeFunctionWRTArcLength(org_x, org_y)

	finalVals = argrelextrema(curvature, np.greater, order=2)

	#print "finalVals"
	#print finalVals

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



	
