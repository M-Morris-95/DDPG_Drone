mdl = 'Tricopter';
%https://harikrishnansuresh.github.io/assets/deep-rl-final.pdf
open_system(mdl)

% Specify observer information
obsInfo = rlNumericSpec([2 1],...
    'LowerLimit',[-1 -1]',...
    'UpperLimit',[ 1  1]');
obsInfo.Name = 'observer';
obsInfo.Description = 'yaw to target, pitch to target, dist to target';
numObs = obsInfo.Dimension(1);

% Specify actor information
actInfo = rlNumericSpec([3 1],...
    'LowerLimit',-1',...
    'UpperLimit', 1');
actInfo.Name = 'actor';
obsInfo.Description = 'Thr, roll, pitch, Yaw';
numAct = actInfo.Dimension(1);

% Initialise environment - simulator, agent, observer information, actor
% information
env = rlSimulinkEnv('Tricopter','Tricopter/RL Agent',...
    obsInfo,actInfo);

% Set model params
Ts = 0.4; % sample time
Tf = 40;  % finish time
rng('shuffle');    % Set rng seed

% Create some convolutional neural nets for the critic
% specify the number of outputs for the hidden layers.
hiddenLayerSize = 6; 

statePath = [
    imageInputLayer([numObs 1 1], 'Normalization', 'none', 'Name', 'observation')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticStateFC1')
    reluLayer('Name', 'CriticRelu1')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticStateFC2')];
actionPath = [
    imageInputLayer([numAct 1 1], 'Normalization', 'none', 'Name', 'action')
    fullyConnectedLayer(hiddenLayerSize, 'Name', 'CriticActionFC1', 'BiasLearnRateFactor', 0)];
commonPath = [
    additionLayer(2,'Name', 'add')
    reluLayer('Name','CriticCommonRelu')
    fullyConnectedLayer(1, 'Name', 'CriticOutput')];

criticNetwork = layerGraph();
criticNetwork = addLayers(criticNetwork,statePath);
criticNetwork = addLayers(criticNetwork,actionPath);
criticNetwork = addLayers(criticNetwork,commonPath);
criticNetwork = connectLayers(criticNetwork,'CriticStateFC2','add/in1');
criticNetwork = connectLayers(criticNetwork,'CriticActionFC1','add/in2');
criticOptions = rlRepresentationOptions('LearnRate',1e-03,'GradientThreshold',1);
critic = rlRepresentation(criticNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'action'},criticOptions);

% Create the actor network (another CNN)
actorNetwork = [
    imageInputLayer([numObs 1 1],'Normalization','none','Name','observation')
    fullyConnectedLayer(hiddenLayerSize,'Name','fc1')
    reluLayer('Name','relu1')
    fullyConnectedLayer(numAct,'Name','fc2')
    tanhLayer('Name','tanh1')];

actorOptions = rlRepresentationOptions('LearnRate',1e-04,'GradientThreshold',1);
actor = rlRepresentation(actorNetwork,obsInfo,actInfo,'Observation',{'observation'},'Action',{'tanh1'},actorOptions);


% Create the DDPG agent
agentOptions = rlDDPGAgentOptions(...
    'SampleTime',Ts  ,...
    'TargetSmoothFactor',0.125,...
    'ExperienceBufferLength',5e3 ,...
    'DiscountFactor',0.98,...
    'MiniBatchSize',32);

agentOpts.NumStepsToLookAhead = 1;
agentOpts.NoiseOptions.InitialAction = 0;
agentOpts.NoiseOptions.Mean = 0;
agentOpts.NoiseOptions.Variance = 0.3;
agentOpts.NoiseOptions.VarianceDecayRate = 0.6;
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
    'StopTrainingValue',8000);

disp('set up complete')
doTraining = true;
if doTraining
    % Train the agent.
    trainingStats = train(agent,env,trainOpts);
else
    % Load pretrained agent.
end
experience = sim(env,agent);
