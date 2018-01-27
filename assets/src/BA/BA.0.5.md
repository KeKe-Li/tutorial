### Deeplearning Algorithms tutorial
谷歌的人工智能位于全球前列，在图像识别、语音识别、无人驾驶等技术上都已经落地。而百度实质意义上扛起了国内的人工智能的大旗，覆盖无人驾驶、智能助手、图像识别等许多层面。苹果业已开始全面拥抱机器学习，新产品进军家庭智能音箱并打造工作站级别Mac。另外，腾讯的深度学习平台Mariana已支持了微信语音识别的语音输入法、语音开放平台、长按语音消息转文本等产品，在微信图像识别中开始应用。全球前十大科技公司全部发力人工智能理论研究和应用的实现，虽然入门艰难，但是一旦入门，高手也就在你的不远处！
AI的开发离不开算法那我们就接下来开始学习算法吧！

#### 局部加权学习算法（LWR）

局部加权回归（LWR）是非参数学习方法。 首先参数学习方法是这样一种方法：在训练完成所有数据后得到一系列训练参数，然后根据训练参数来预测新样本的值，这时不再依赖之前的训练数据了，参数值是确定的。而非参数学习方法是这样一种算法：在预测新样本值时候每次都会重新训练数据得到新的参数值，也就是说每次预测新样本都会依赖训练数据集合，所以每次得到的参数值是不确定的。
局部加权回归（LWR）是我们遇到的第一个non-parametric（非参数）学习算法，而线性回归则是我们遇到的以一个parametric（参数）学习算法。因为参数学习算法它有固定的明确的参数，所以参数一旦确定，就不会改变了，我们不需要在保留训练集中的训练样本。而非参数学习算法，每进行一次预测，就需要重新学习一组，是变化的，所以需要一直保留训练样本。因而，当训练集的容量较大时，非参数学习算法需要占用更多的存储空间，计算速度也较慢。所以有得必有失，效果好了，计算速度却降下来了。

```python

using namespace std;  
  
const int Number = 6;  
const int Dimesion = 3;  
const float learningRate=0.001;       
const float errorThr=1; //variance threshold  
const int MAX=1000;     //Max times of iteration  
  
typedef struct Data{  
    float vectorComponent[Dimesion];  
}vectorData;  
  
vectorData x[Number] = {  
    {1,1,8},  
    {1,1,3},  
    {1,1,6},  
    {1,2,3},  
    {1,2,1},  
    {1,2,2},  
};  
float y[Number]={2,10,5,13,5,8};  
/lwr(局部线性回归)  
float weightValue(vectorData xi,vectorData x){  
    float weight = 0.0;  
    for(int i=0;i<Dimesion;i++){  
        weight+=pow(xi.vectorComponent[i]-x.vectorComponent[i],2);  
    }  
    float tempWeight = exp(-(weight/(2*36)));  
    if(tempWeight<0.02)  
        tempWeight = 0.0;  
    return tempWeight;  
}  
  
float multiPly(vectorData x1,vectorData x2){  
    float temp = 0.0;  
    for(int i=0;i<Dimesion;i++){  
        temp += x1.vectorComponent[i]*x2.vectorComponent[i];  
    }  
    return temp;  
}  
  
vectorData addVectorData(vectorData x1,vectorData x2){  
    vectorData temp;  
    for(int i=0;i<Dimesion;i++)  
        temp.vectorComponent[i] = x1.vectorComponent[i]+x2.vectorComponent[i];  
    return temp;  
}  
  
vectorData minusVectorData(vectorData x1,vectorData x2){  
    vectorData temp;  
    for(int i=0;i<Dimesion;i++)  
        temp.vectorComponent[i] = x1.vectorComponent[i]-x2.vectorComponent[i];  
    return temp;  
}  
  
vectorData numberMultiVectorData(float para,vectorData x1){  
    vectorData temp;  
    for(int i=0;i<Dimesion;i++)  
        temp.vectorComponent[i] = x1.vectorComponent[i]*para;  
    return temp;  
}  
float costFunction(vectorData parameter[],vectorData inputData[],float inputResultData[],vectorData object){  
    float costValue = 0.0;  
    float tempValue = 0.0;  
    float weightedValue = 0.0;  
    for(int i=0;i<Number;i++){  
        tempValue = 0.0;  
          
        //consider all the parameters although most of them is zero  
        for(int j=0;j<Number;j++)  
            tempValue += multiPly(parameter[j],inputData[i]);  
        costValue += weightValue(inputData[i],object)*pow((inputResultData[i]-tempValue),2);      
    }  
  
    return (costValue/2*4);  
}  
  
  
int LocallyWeightedAgression(vectorData parameter[],vectorData inputData[],float resultData[],vectorData objectVector){  
    float tempValue = 0.0;  
    float errorCost = 0.0;  
    float weightedValue = 0.0;  
    errorCost=costFunction(parameter,inputData,resultData,objectVector);  
    if(errorCost<errorThr)  
        return 1;  
    for(int iteration=0;iteration<MAX;iteration++){  
  
        //stochastic  
        for(int i=0;i<Number;i++){  
            //calculate the h(x)  
            weightedValue = weightValue(inputData[i],objectVector);  
            tempValue=0.0;  
            for(int j=0;j<Number;j++)  
                tempValue+=multiPly(parameter[j],inputData[i]);  
            //update the parameter by stochastic(随机梯度下降)  
            printf("the next parameter is ");  
            for(int ii=0;ii<Number;ii++){  
                parameter[ii] = addVectorData(parameter[ii],numberMultiVectorData(weightedValue*learningRate*(resultData[i]-tempValue),inputData[i]));  
                if(multiPly(parameter[ii],parameter[ii])!=0){  
                    for(int jj=0;jj<Dimesion;jj++){  
                        printf("%f ",parameter[ii].vectorComponent[jj]);  
                    }  
                }  
            }  
            printf("\n");  
            errorCost=costFunction(parameter,inputData,resultData,objectVector);  
            printf("error cost is %f\n",errorCost);  
            if(errorCost<errorThr)  
                break;  
        }//end stochastic one time  
  
    }//end when the iteration becomes MAX   
  
    //calculate the object vector  
    float resultValue = 0.0;  
    for(int i=0;i<Number;i++){  
        resultValue += weightValue(inputData[i],objectVector)*multiPly(parameter[i],objectVector);  
    }  
    printf("result value is %f \n",resultValue);  
    return 1;  
}  
  
int testLWA(){  
    vectorData objectData = {1,1.5,1.5};  
    vectorData localParameter[Number] = {0.0};  
    LocallyWeightedAgression(localParameter,x,y,objectData);  
    return 1;  
}  
int main(){  
  //  DescendAlgorithm(parameter,x,y);  
    //clearParameter(parameter);  
    //Stochastic(parameter,x,y);  
    //float ForTestData[] = {1,10,20};  
    //testData(ForTestData);  
    testLWA();  
    system("pause");  
    return 1;  
}  
```
