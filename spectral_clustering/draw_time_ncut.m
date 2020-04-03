data_names = ["segment_uni","MnistData_05_uni","MnistData_10_uni",...
    "isolet5","USPSdata_uni","letter-recognition"];
abbr_data_names = ["D1","D2","D3","D4","D5","D6"];
algorithms = ["wkk","sample_wkk"];
num_points = [2310,3495,6996,7797,9298,20000];
% data_type = "running_time"; % running_time/obj
data_type = "Ncuts";
data = zeros(length(algorithms),length(data_names));
for i = 1:length(data_names)
    for j = 1:length(algorithms)
        nr = num2str(sqrt(num_points(i)));
        v_struct = load(join([data_names(i),algorithms(j),'.mat'],"_"));
        value = getfield(v_struct,data_type);
        data(j,i) = mean(value);
    end
end
f = figure();
is_bar = true;
if is_bar
    c_data_names = categorical(abbr_data_names);
    c_data_names = reordercats(c_data_names,cellstr(abbr_data_names));
    bar(c_data_names,data');
    legend('wkk','sample\_wkk','Location','best');
else
    % data' because it has the same first dim with num_points
    p = plot(num_points,data','LineWidth',1,'MarkerSize',4);
    set(p,{'color'},{'b';'r'});
    set(p,{'Marker'},{'o';'v'});
    set(p,{'MarkerFaceColor'},{'b';'r'});

    legend('wkk','sample\_wkk','Location','best');
    xlabel('number of points');
    if strcmp(data_type,"running_time")
        ylabel('time(s)');
    elseif strcmp(data_type,"Ncuts")
        ylabel('Ncut');
    end
end

% saveas(f,strcat(data_type,'.jpg')); % image-running-time/ncut