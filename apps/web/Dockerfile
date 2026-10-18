# 浏览器端静态站点（air-gapped 部署用）。
#
# 模型权重不打进镜像：构建时 .dockerignore 排除 apps/web/public/models，
# 运行时由 compose 只读挂载到 /usr/share/nginx/html/models。
# 这样同一个镜像可以配不同量化版本，也不会因 1GB+ 权重产生巨型镜像层。
FROM node:22-bookworm-slim AS build

WORKDIR /app
# 依赖层单独缓存：只改源码时不必重装 node_modules
COPY apps/web/package.json apps/web/package-lock.json ./
RUN npm ci

COPY apps/web/ ./
# vite.config.ts 以 apps/web 为基准解析 ../../packages/eval（评测资产唯一真源）
COPY packages/eval /packages/eval
RUN npm run build

FROM nginx:1.27-alpine
COPY apps/web/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html
EXPOSE 80
