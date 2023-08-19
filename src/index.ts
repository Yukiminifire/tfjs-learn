import {tensor,scalar, variable, rand, tensor1d, SGDOptimizer, train} from "@tensorflow/tfjs"
import "@tensorflow/tfjs-node"

// y = 2x + 3
const dataX = tensor([1,2,3,4,5,6,7,8,9])
const dataY = dataX.mul(scalar(2)).add(scalar(3))

// 尝试拟合
const k = variable(scalar(Math.random()))
const b = variable(scalar(Math.random()))

// 优化器
const opt = train.sgd(0.01)

// 训练
for (let index = 0; index < 10000; index++) {
    opt.minimize(()=>{
        const perdict = k.mul(dataX).add(b)
        const loss = perdict.sub(dataY).square().mean()
        // loss.print()
        return loss.asScalar()
    })
}

// 训练结果
console.log("result")
k.print()
b.print()