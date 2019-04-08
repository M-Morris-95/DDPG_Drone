mdl = 'Tricopter';
open_system(mdl)

% Specify observer information
obsInfo = rlNumericSpec([6 1],...
    'LowerLimit',[-1 -1 -1 -1 -1 -1]',...
    'UpperLimit',[ 1  1  1  1  1  1]');
obsInfo.Name = 'observer';
obsInfo.Description = 'yaw to target, pitch to target, dist to target';
numObs = obsInfo.Dimension(1);

% Specify actor information
actInfo = rlNumericSpec([3 1],...
    'LowerLimit',[-1 -1 -1]',...
    'UpperLimit',[ 1  1  1]');
actInfo.Name = 'actor';
obsInfo.Description = 'Xv,Yv,Zv';
numAct = actInfo.Dimension(1);


% Initialise environment - simulator, agent, observer information, actor
% information
env = rlSimulinkEnv('Tricopter','Tricopter/RL Agent',...
    obsInfo,actInfo);

% Set model params
Ts = 0.4; % sample time
Tf = 80;  % finish time
rng('shuffle');    % Set rng seed

% Create some convolutional neural nets for the critic
% specify the number of outputs for the hidden layers.
hiddenLayerSize = 128; 

observationPath = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')
    additionLayer(2,'Name','add')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(1,'Name','fc4')];
actionPath = [
    imageInputLayer([numAct 1 1],'Normalization','none','Name','action')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc5')];

% create the layerGraph
criticNetwork = layerGraph(observationPath);
criticNetwork = addLayers(criticNetwork,actionPath);

% connect actionPath to obervationPath
criticNetwork = connectLayers(criticNetwork,'fc5','add/in2');
criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1,'UseDevice',"gpu");
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOptions);




% figure
% plot(criticNetwork)

% Create the actor network (another CNN)
actorNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc2')
    reluLayer('Name','relu2')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc3')
    reluLayer('Name','relu3')
    fullyConnectedLayer(numAct,'Name','fc4')
    tanhLayer('Name','tanh1')];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1,'UseDevice',"gpu");

actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'tanh1'},actorOptions);


% Create the DDPG agent
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts  ,...
    'TargetSmoothFactor',1e-3,...
    'ExperienceBufferLength',1e6 ,...
    'DiscountFactor',0.98,...
    'MiniBatchSize',64);

agentOpts.NumStepsToLookAhead = 16;
agentOpts.NoiseOptions.InitialAction = 0;
agentOpts.NoiseOptions.Mean = 0;
agentOpts.NoiseOptions.Variance = 1e-3;
agentOpts.NoiseOptions.VarianceDecayRate = 1e-2;

agent = rlDDPGAgent(actor,critic,agentOptions);


maxepisodes = 100000;
maxsteps = ceil(Tf/Ts);
trainOpts = rlTrainingOptions(...
    'MaxEpisodes',maxepisodes, ...
    'MaxStepsPerEpisode',maxsteps, ...
    'ScoreAveragingWindowLength',64, ...
    'Verbose', false, ...
    'Plots','training-progress',...
    'StopTrainingCriteria','AverageReward',...
    'StopTrainingValue',800);

disp('set up complete')
doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent.
end
experience = sim(env,agent);
