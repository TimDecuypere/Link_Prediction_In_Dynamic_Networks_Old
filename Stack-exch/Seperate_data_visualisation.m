clear all
beep off
close all

%%% Data prep employees %%%

filename='ia-stackexch-user-marks-post-und.txt';
M=readtable(filename);
A = round(table2array(M));
B = A(1:1237867,:); %% select range (manually found indices)
dlmwrite('stack-exch-oct2008-2011.txt',B, 'precision','%.f')

dt = datetime( B(:,4), 'ConvertFrom', 'posixtime' );
yMonth = discretize(dt,'month','categorical');
yWeek = discretize(dt,'week','categorical');
%%%% Find month indices
index_start=1;
m_index = [];
i=1;
while (yMonth(index_start) ~= yMonth(length(yMonth)))
    m_index = [m_index, find(yMonth > yMonth(index_start),1)];
    index_start = m_index(i);
    i=i+1;
end


m16 = B(1:(m_index(6)-1),:);
m712 = B(m_index(6):(m_index(12)-1),:);
m1318 = B(m_index(12):(m_index(18)-1),:);
m1924 = B(m_index(18):(m_index(24)-1),:);
m2530 = B(m_index(24):(m_index(30)-1),:);
m3136 = B(m_index(30):(length(B)),:);
mStat = B(1:(m_index(30) - 1),:);


dlmwrite('stack-exch-m16.txt',m16, 'precision','%.f')
dlmwrite('stack-exch-m712.txt',m712, 'precision','%.f')
dlmwrite('stack-exch-m1318.txt',m1318, 'precision','%.f')
dlmwrite('stack-exch-m1924.txt',m1924, 'precision','%.f')
dlmwrite('stack-exch-m2530.txt',m2530, 'precision','%.f')
dlmwrite('stack-exch-m3136.txt',m3136, 'precision','%.f')
dlmwrite('stack-exch-mStat.txt', mStat, 'precision', '%.f')

% figure
% histogram(yMonth)
% xlabel('Month histogram facebook forum');
% 
% figure
% histogram(yWeek);
% xlabel('Week histogram facebook forum');
