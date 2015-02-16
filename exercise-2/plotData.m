%ml-008 Exercise2
%Based off of sample code provided by coursera Machine Learning Course
%ml-008 taught by Andrew NG of Stanford
%@author Adam Tetelman 2/15/2015
function plotData(X, y)
% Create New Figure
figure; hold on;


%Parse out the positive/negative samples
positive = find(y==1); negative = find(y==0);

%Plot x(m,1), x(m,2) use + for y=1 and circe for y=0
plot(X(positive,1),X(positive,2),'k+', 'Markersize', 7, ...
    'MarkerEdgeColor', 'b')
plot(X(negative,1),X(negative,2),'ko', 'Markersize', 7, ...
    'MarkerFaceColor', 'y')

%Set labels
ylabel('X(:,2)'); xlabel('X(:,1)'); 

hold off;

end
