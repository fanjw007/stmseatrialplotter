"""
_________________________________

VESSEL MANOEUVRING ANALYSIS
TURNING CICLE (PORT)
_________________________________

Coded by: Fan Jun Wei

Last Updated: 05.04.2021
"""

# -----------------------------------------
# Libraries
# IMPORT REQUIRED LIBRARIES FOR SCRIPT
# -----------------------------------------
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import math
import pandas as pd
from jinja2 import Environment, FileSystemLoader

# -----------------------------------------
# Input
# DEFINE PROJECT VARIABLES
# -----------------------------------------
project_code = "BO0683"
ship_length = 23

# -----------------------------------------
# Function
# INTERPLOATE VARIABLE FROM HEADING INPUT
# -----------------------------------------
def interpolateVar(var_lb,var_ub,head_lb,head_ub,target_head):
    
    target_var = ((var_ub-var_lb)*((target_head-head_lb)/(head_ub-head_lb)))+var_lb
    return target_var

# ------------------------------------------
# Function
# GET DESIRED VARIABLE DATA FROM HEADING
# -----------------------------------------
def getVar(var_dict,head_dict,i_head):
    
    looper = True                 # initialise Boolean variable for looping
    
    # loop through head_dict for heading of interest
    while looper is True:
        for pos in head_dict:
            if head_dict[pos]<=i_head:
                head_lb = pos
            else:
                head_ub = pos
                looper = False
                break
    
    # get variable at which vessel hits desired heading
    target_var=interpolateVar(var_dict[head_lb],var_dict[head_ub],head_dict[head_lb],head_dict[head_ub],i_head)
    return target_var

# ------------------------------------------
# Function
# GET STEADY TURNING DIAMETER
# -----------------------------------------
def steadyDia(northing_dict,easting_dict,head_dict,trim_head):
    
    # initialise summation storage
    northing_sum = 0
    easting_sum= 0
    head_count = 0
    
    for pos in head_dict:
        if head_dict[pos] >= trim_head and head_dict[pos] <= trim_head+360:
            northing_sum += northing_dict[pos]
            easting_sum += easting_dict[pos]
            head_count +=1
    
    northing_avg = northing_sum/head_count
    easting_avg = easting_sum/head_count
    
    return northing_avg, easting_avg

# ------------------------------------------
# Function
# GET TANGENT ANGLE FOR DRIFT ANGLE
# -----------------------------------------
def tangentAngle(northing_dict,easting_dict,head_dict,tangent_head):
       
    looper = True                   # initialise Boolean variable for looping
    
    while looper is True:
        for pos in head_dict:
            if head_dict[pos] <= tangent_head:
                head_lb = pos
            else:
                looper = False
                break
    
    theta = math.atan((northing_dict[head_lb]-northing_dict[head_lb-1])/\
                      (easting_dict[head_lb]-easting_dict[head_lb-1]))
    
    return math.degrees(theta)
            
# ------------------------
# Function
# MAIN
# ------------------------
#def main():
    
# Get Input File
#raw_data = open('621_R5_TurningCirclePort_RawData.csv','r')
raw_data = open('683_R20_TurningCirclePort_RawData.csv','r')

# Set Up Data Lists
pos_list = []
time_dict = {}
gyro_dict = {}
speed_dict = {}
rot_dict = {}
dir_dict = {}
wind_dict = {}
easting_dict = {}
northing_dict = {}

# Get Header
header = raw_data.readline().replace('\n', '').split(",")

# Read RAW DATA and store to dictionaries
for line in raw_data:
    extract = line.replace('\n', '').split(",")
    pos_list.append(int(extract[0]))
    try:
        time_dict[int(extract[0])] = datetime.strptime(extract[1],"%H:%M:%S.%f")
    except ValueError:
        time_dict[int(extract[0])] = datetime.strptime(extract[1],"%M:%S.%f")
    gyro_dict[int(extract[0])] = float(extract[2])
    speed_dict[int(extract[0])] = float(extract[3])
    rot_dict[int(extract[0])] = float(extract[4])
    dir_dict[int(extract[0])] = float(extract[5])
    wind_dict[int(extract[0])] = float(extract[6])
    easting_dict[int(extract[0])] = float(extract[7])
    northing_dict[int(extract[0])] = float(extract[8])

# ---------------------------
# Data Processing
# GENERAL DATA READING
# ---------------------------

initial_speed = speed_dict[pos_list[0]]

# ---------------------------
# Data Processing
# SPEED TIME ANALYSIS
# ---------------------------

# Convert Time to Seconds

elapsed_time = 0                            # initialise elapsed time

for pos in pos_list:                        # loop through entire data list
    if pos!=pos_list[0]:                    # check if first position
        dt = time_dict[pos] - prev_time     # get time difference
        elapsed_time += dt.total_seconds()  # add time diff to elapsed time
    prev_time = time_dict[pos]              # update old time
    time_dict[pos] = elapsed_time           # update stored value to seconds

# ---------------------------
# Data Processing
# TURNING CIRCLE ANALYSIS
# ---------------------------

beta_0 = math.radians(gyro_dict[pos_list[0]])   # initial heading
initial_heading = gyro_dict[pos_list[0]]        # initial heading
head_dict = {}                                  # initialise heading dictionary

# Initialise travel variables
ng_trav=0                                           
eg_trav=0
new_heading = 0

# zero gyro values based on initial heading
corr_gyro_dict = {k:v - initial_heading for k,v in gyro_dict.items()}


for pos in pos_list:                            # loop through entire data list
    if pos!=pos_list[0]:                        # if not first data point
        # get change in northing and easting    
        delta_n = northing_dict[pos] - prev_n
        delta_e = easting_dict[pos] - prev_e 
        
        # map change to new coordinate system
        delta_ng = delta_n*math.cos(beta_0) + delta_e*math.sin(beta_0)
        delta_eg = -delta_n*math.sin(beta_0) + delta_e*math.cos(beta_0)
        
        # add to travel variables
        ng_trav += delta_ng
        eg_trav += delta_eg
        
        # compute heading variable
        delta_heading = prev_heading - corr_gyro_dict[pos]
        if abs(delta_heading) > 180:                 # catch reset errors
            new_heading += 360
        new_heading += delta_heading
        if new_heading < 0:                     # catch reset errors
            new_heading += 360
    
    # update previous positions for storage
    prev_n = northing_dict[pos]
    prev_e = easting_dict[pos]
    prev_heading = corr_gyro_dict[pos]
    
    # push transformed coordinate variables into dictionary
    northing_dict[pos] = ng_trav
    easting_dict[pos] = abs(eg_trav)
    head_dict[pos] = new_heading

# get time data
time_090 = getVar(time_dict, head_dict, 90)
time_180 = getVar(time_dict, head_dict, 180)
time_270 = getVar(time_dict, head_dict, 270)
time_360 = getVar(time_dict, head_dict, 360)
turn_rate = (360/time_360)*60

# get pos data
pos_n_090 = getVar(northing_dict,head_dict, 90)
pos_e_090 = getVar(easting_dict,head_dict, 90)
pos_n_180 = getVar(northing_dict,head_dict, 180)
pos_e_180 = getVar(easting_dict,head_dict, 180)
pos_n_270 = getVar(northing_dict,head_dict, 270)
pos_e_270 = getVar(easting_dict,head_dict, 270)
pos_n_360 = getVar(northing_dict,head_dict, 360)
pos_e_360 = getVar(easting_dict,head_dict, 360)

# get track reach at 10 degree
track_reach_010 = getVar(northing_dict,head_dict, 10)
# get advance
advance = getVar(northing_dict,head_dict, 90)
# get transfer
transfer = getVar(easting_dict,head_dict, 90)
# get tactical diameter
tact_dia = getVar(easting_dict,head_dict, 180)

# get steady turning diameter
cent_n, cent_e = steadyDia(northing_dict, easting_dict, head_dict, 270)
opp_n = cent_n + abs(pos_n_360-cent_n)      # positional data for plotting
opp_e = cent_e - abs(pos_e_360-cent_e)      # positional data for plotting
steady_turn_dia = math.sqrt(((pos_n_360-cent_n)**2)+((pos_e_360-cent_e)**2))*2

# get drift angle
theta = tangentAngle(northing_dict, easting_dict, head_dict, 360)
drift_angle = theta + 90

# print turning circle data to command line
print('{:23}: {:<8.2f}{:9}'.format("Approaching Course",initial_heading,"deg"))
print('{:23}: {:<8.2f}{:9}'.format("Approaching Speed",initial_speed,"knots"))
print('{:23}: {:<8.2f}{:9}'.format("Track Reach at 10 deg",track_reach_010,"m"))
print('{:23}: {:<8.2f}{:9}'.format("Advance",advance,"m"))
print('{:23}: {:<8.2f}{:9}'.format("Transfer",transfer,"m"))
print('{:23}: {:<8.2f}{:9}'.format("Tactical Diameter",tact_dia,"m"))
print('{:23}: {:<8.2f}{:9}'.format("Steady Diameter",steady_turn_dia,"m"))
print('{:23}: {:<8.2f}{:9}'.format("Time Taken (090 deg)",time_090,"sec"))
print('{:23}: {:<8.2f}{:9}'.format("Time Taken (180 deg)",time_180,"sec"))
print('{:23}: {:<8.2f}{:9}'.format("Time Taken (270 deg)",time_270,"sec"))
print('{:23}: {:<8.2f}{:9}'.format("Time Taken (360 deg)",time_360,"sec"))
print('{:23}: {:<8.2f}{:9}'.format("Rate of Turn",turn_rate,"deg/min"))
print('{:23}: {:<8.2f}{:9}'.format("Drift Angle",drift_angle,"deg"))

# ---------------------------
# Data Processing
# SPEED-TIME GRAPH
# ---------------------------

# create graph
st_graph = plt.figure(figsize=(13,4))

# compute graph limits
st_graph_ymax = 2*math.ceil(max(speed_dict.values())/2)
st_graph_ymin = 2*math.floor(min(speed_dict.values())/2)

# plot graph
st_axes = plt.axes()
st_axes.plot(time_dict.values(),speed_dict.values())

# set plot limits
st_axes.set_ylim(st_graph_ymin,st_graph_ymax)
st_axes.set_xlim(0,elapsed_time)

# set up title and labels
st_axes.set_title("Speed-Time Plot")
st_axes.set_xlabel("Time (s)")
st_axes.set_ylabel("Speed (knots)")

# set axis tick frequency
st_axes.set_xticks(np.arange(0,elapsed_time,5))
st_axes.set_yticks(np.arange(st_graph_ymin,st_graph_ymax,2))

# set grid
st_axes.grid(True)

# save plot
st_graph.savefig('figs/SpeedTimePlot.svg', bbox_inches='tight', pad_inches=0, format='svg')
st_graph.savefig('figs/SpeedTimePlot.png', dpi=300,bbox_inches='tight', format='png')


# ---------------------------
# Data Processing
# TURNING CIRCLE PLOT
# ---------------------------

# create graph
turn_graph = plt.figure(figsize=(10,12))

# compute graph limits
turn_graph_xmax = 100*math.ceil(math.ceil(max(easting_dict.values()))/100)+150
turn_graph_xmin = 100*math.ceil(math.floor(min(easting_dict.values()))/100)-150
turn_graph_ymax = 100*math.ceil(math.ceil(max(northing_dict.values()))/100)+100
turn_graph_ymin = 100*math.ceil(math.floor(min(northing_dict.values()))/100)-150

# plot graph
turn_axes = plt.axes()
turn_axes.plot(easting_dict.values(),northing_dict.values(),zorder=0.9)

# set plot limits
turn_axes.set_xlim(turn_graph_xmax,turn_graph_xmin)
turn_axes.set_ylim(turn_graph_ymin,turn_graph_ymax)

xticks_list = np.arange(turn_graph_xmax,turn_graph_xmin-1,-50)
yticks_list = np.arange(turn_graph_ymin,turn_graph_ymax+1,50)

# set axis tick frequency
turn_axes.set_xticks(xticks_list)
turn_axes.set_yticks(yticks_list)

# append units to axis ticks
turn_axes.xaxis.set_major_formatter(FormatStrFormatter('%d m'))
turn_axes.yaxis.set_major_formatter(FormatStrFormatter('%d m'))

# reformat axis ticks
turn_axes.tick_params(axis="x",direction="in", pad=-50, labelrotation=90)
turn_axes.tick_params(axis="y",direction="in", pad=-50)
plt.setp(turn_axes.get_xticklabels()[0], visible=False)
plt.setp(turn_axes.get_xticklabels()[-1], visible=False)
plt.setp(turn_axes.get_yticklabels()[0], visible=False)
plt.setp(turn_axes.get_yticklabels()[-1], visible=False)

# set grid
turn_axes.set_aspect('equal')
turn_axes.grid(True)
turn_axes.set_axisbelow(True)

# Plot Turning Circle Heading Arrow
turn_axes.arrow(turn_graph_xmin+25, turn_graph_ymax-30, 0, 15, head_width=5, head_length=5, fc='k', ec='k')
turn_axes.text(turn_graph_xmin+32, turn_graph_ymax-40,str(initial_heading)+'\u00B0')

# Plot Advance
turn_axes.annotate("", \
                   xy=(-50,northing_dict[pos_list[0]]),xycoords='data', \
                   xytext=(-50,advance),textcoords='data', \
                   arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5', \
                                   shrinkA=0,shrinkB=0,color='midnightblue'))
turn_axes.plot([-45,pos_e_090],[pos_n_090,pos_n_090],'--',color='lightgrey',\
               zorder=0,lw=1)
turn_axes.text(-55,advance/2,'Advance\n{:.3f} m\n({:.3f} L)'.\
               format(advance,advance/ship_length),color='midnightblue')

# Plot Transfer
turn_axes.annotate("", \
                   xy=(0,turn_graph_ymax-80),xycoords='data', \
                   xytext=(transfer,turn_graph_ymax-80),textcoords='data', \
                   arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5', \
                                   shrinkA=0,shrinkB=0,color='midnightblue'))
turn_axes.plot([pos_e_090,pos_e_090],[pos_n_090,turn_graph_ymax-85],'--',color='lightgrey',\
               zorder=0,lw=1)
turn_axes.text(transfer/2,turn_graph_ymax-72,'Transfer\n{:.3f} m\n({:.3f} L)'.\
               format(transfer, transfer/ship_length), \
               horizontalalignment='center',color='midnightblue')
    
# Plot Tactical Diameter
turn_axes.annotate("", \
                   xy=(0,-25),xycoords='data', \
                   xytext=(tact_dia,-25),textcoords='data', \
                   arrowprops=dict(arrowstyle='|-|, widthA=0.5, widthB=0.5', \
                                   shrinkA=0,shrinkB=0,color='midnightblue'))
turn_axes.plot([pos_e_180,pos_e_180],[-20,pos_n_180],'--',color='lightgrey',\
               zorder=0,lw=1)
turn_axes.text(tact_dia/2,-50,'Tactical Diameter\n{:.3f} m\n({:.3f} L)'.\
               format(tact_dia,tact_dia/ship_length), \
               horizontalalignment='center',color='midnightblue')
    
# Plot Steady Turning Diameter
turn_axes.annotate("", \
                   xy=(pos_e_360,pos_n_360),xycoords='data', \
                   xytext=(opp_e,opp_n),textcoords='data', \
                   arrowprops=dict(arrowstyle='|-|, widthA=0.25, widthB=0.25', \
                                   shrinkA=1.0,shrinkB=0,color='midnightblue'),\
                   zorder=1.0)
turn_axes.text(cent_e-3,cent_n+1,'Steady\nDiameter\n{:.3f} m\n({:.3f} L)'.\
               format(steady_turn_dia,steady_turn_dia/ship_length), \
               horizontalalignment='left',verticalalignment='top',\
               size=7,color='midnightblue')
    
# Plot Positional Data
turn_axes.plot(pos_e_090,pos_n_090,'x',c='maroon',zorder=2.0)
turn_axes.text(pos_e_090-5,pos_n_090+5,'90\u00B0 @ {:.3f} s'.format(time_090),c='maroon',size=12)
turn_axes.plot(pos_e_180,pos_n_180,'x',c='maroon',zorder=2.0)
turn_axes.text(pos_e_180-5,pos_n_180,'180\u00B0 @ {:.3f} s'.format(time_180),c='maroon',size=12)
turn_axes.plot(pos_e_270,pos_n_270,'x',c='maroon',zorder=2.0)
turn_axes.text(pos_e_270+75,pos_n_270,'270\u00B0 @ {:.3f} s'.format(time_270),c='maroon',size=12)
turn_axes.plot(pos_e_360,pos_n_360,'x',c='maroon',zorder=2.0)
turn_axes.text(pos_e_360+75,pos_n_360-7,'360\u00B0 @ {:.3f} s'.format(time_360),c='maroon',size=12)

# Plot Centre
turn_axes.plot(cent_e,cent_n,'o',c='aqua',zorder=2.0,markersize=3)
                 
# Save Figure
turn_graph.savefig('figs/TurningCirclePlot.svg', bbox_inches='tight', pad_inches=0, format='svg')
turn_graph.savefig('figs/TurningCirclePlot.png', dpi=300,bbox_inches='tight', format='png')

# ------------------------
# Reporting
# GENERATE REPORT
# ------------------------

# set up environment
env = Environment(loader=FileSystemLoader('.'))
template = env.get_template("turningCircleReportGeneration_v2.html")

# set input variables
template_vars = {"project_code": project_code,
                 "title" : project_code+" TURNING CIRCLE REPORT",
                 "turning_circle_plot": "figs/TurningCirclePlot.svg",
                 "speed_time_plot": "figs/SpeedTimePlot.svg"}

# create input dataframe
df = pd.DataFrame([initial_heading,initial_speed,track_reach_010,turn_rate,
                   drift_angle],
                  ["Approaching Course","Approaching Speed","Track Reach at 10 deg",
                   "Rate of Turn","Drift Angle"])
df_rounded = round(df,3)

# render html file
html_out = template.render(template_vars,
                           datatable = df_rounded.to_html(header=False))

# export to HTML
with open('turningCiclePort_Report.html', 'w') as f:
    f.write(html_out)

# ------------------------
# Program
# RUN
# ------------------------
#main()