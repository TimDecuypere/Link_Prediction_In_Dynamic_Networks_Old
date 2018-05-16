clear all
beep off
close all

filename='ia-stackexch-user-marks-post-und.txt';
M=readtable(filename);
A = table2array(M);
dt = datetime( round(A(:,4)), 'ConvertFrom', 'posixtime' );

yMonth = discretize(dt,'month','categorical');
figure
histogram(yMonth);
xlabel('Month histogram -stackexch-user-marks-post-und');

yWeek = discretize(dt,'week','categorical');
figure
histogram(yWeek);
xlabel('Week histogram -stackexch-user-marks-post-und');