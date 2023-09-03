import {
  tensor,
  scalar,
  variable,
  train,
  sequential,
  layers,
  losses,
  browser,
} from "@tensorflow/tfjs";
import { Point2D, render, show } from "@tensorflow/tfjs-vis";
import "@tensorflow/tfjs-backend-webgl";

// y = 2x + 3
const dataX = tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]);
const dataY = dataX.mul(scalar(2)).add(scalar(3));


render.scatterplot(
  { name: "线性回归训练集" },
  {
    values: [1, 2, 3, 4, 5, 6, 7, 8, 9].map((x) => {
      return {
        x,
        y: 2 * x + 3,
      } as Point2D;
    }),
    series: ["默认数据"],
  }
);

// 优化器
const opt = train.sgd(0.02);

const model = sequential();
model.add(
  layers.dense({
    units: 1,
    inputShape: [1],
  })
);

model.compile({
  loss: losses.meanSquaredError,
  optimizer: opt,
});

// 训练
await model.fit(dataX, dataY, {
  batchSize: 8,
  epochs: 160,
  callbacks: show.fitCallbacks({ name: "训练" }, ["loss"]),
});

console.log("weight");
model.getWeights().forEach((e) => {
  e.print();
});
