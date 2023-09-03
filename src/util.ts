import { browser } from "@tensorflow/tfjs";
export async function getImageTensor(url: string) {
  const image = new Image();
  image.src = url;

  await image.decode();
  const imageTensor = browser.fromPixels(image);
  console.log("image", image);
  imageTensor.print();

  return imageTensor;
}
