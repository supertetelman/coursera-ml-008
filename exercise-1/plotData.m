function plotData(x, y)

figure; % open a new figure window

%plot x,y  as red crosses with a size of 10
plot(x,y, 'rx', 'Markersize', 10); 

%set labels
ylabel('Profit in $10,000s'); 
xlabel('Population of City in 10,000s');

end