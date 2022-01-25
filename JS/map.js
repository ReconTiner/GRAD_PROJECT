require([
    "esri/config",
    "esri/Map",
    "esri/views/MapView",
    "esri/layers/FeatureLayer",
    "esri/layers/CSVLayer",
    "esri/widgets/ScaleBar",
    "esri/widgets/Sketch",
    "esri/Graphic",
    "esri/layers/GraphicsLayer",
    "esri/geometry/geometryEngine",],

    function (
        esriConfig,
        Map,
        MapView,
        FeatureLayer,
        CSVLayer,
        ScaleBar,
        Sketch,
        Graphic,
        GraphicsLayer,
        geometryEngine) {

        esriConfig.apiKey = "AAPKf467f63bec014c8fa266ae623ee9f475ecFFIrG6H2k1U_gS-9Eygn9BggsmVW7pPxUaYfiUVSmMcTtFGjEvIqa59lIsHACP";

        // 底图
        const map = new Map({
            basemap: "arcgis-imagery" // Basemap layer service
        });

        // 视图
        const view = new MapView({
            map: map,
            center: [117.5526377222, 34.3100582222],
            zoom: 18,
            container: "main-box"
        });

        // 弹窗
        const popupSamplePoint = {
            "title": "区域相关信息",
            dockEnabled: true,
            dockOptions: {
                breakpoint: false,
                position: "top-right"
            },
            content: [
                {
                    type: "text",
                    text: "<b>影像名称：</b>{ImageName} <br> <b>拍摄时间：</b>{ImageDataTime}"
                },
                {
                    type: "media",
                    mediaInfos: [
                        {
                            title: "<b>病虫害程度统计图</b>",
                            type: "pie-chart",
                            caption: "<b>Health：</b>健康 <br> <b>Normal：</b>一般 <br> <b>Badly：</b>严重",
                            value: {
                                fields: ["Health", "Normal", "Badly"],
                            }
                        },
                        {
                            title: "<b>区域影像</b>",
                            type: "image",
                            value: {
                                sourceURL: "../images/ImageBase/{ImageName}"
                            }
                        },
                    ]
                },
            ]
        };

        // 图层渲染器
        let pointRenderer = {
            type: "simple",
            symbol: {
                type: "simple-marker",
                size: 6,
                color: "blue",
                outline: {
                    width: 0.5,
                    color: "white"
                }
            }
        };

        // // 加载数据图层
        // const sampleLayer = new FeatureLayer({
        //     url: "https://services.arcgis.com/6Kf3sBPu8GLS9ITA/arcgis/rest/services/sample/FeatureServer/0",
        //     outFields: ["ImageName", "ImageDataTime", "Health", "Normal", "Badly"],
        //     popupTemplate: popupSamplePoint,
        //     //renderer: pointRenderer
        //   });

        // 以CSV文件方式加载数据图层
        const sampleLayer = new CSVLayer({
            url: "../sample.csv",
            outFields: ["ImageName", "ImageDataTime", "Health", "Normal", "Badly"],
            popupTemplate: popupSamplePoint,
            renderer: pointRenderer
        });
        map.add(sampleLayer);

        // 添加比例尺工具
        const scalebar = new ScaleBar({
            view: view,
            unit: "metric"
        });
        view.ui.add(scalebar, "bottom-right");

        // 添加工具图层
        const graphicsLayer = new GraphicsLayer();
        map.add(graphicsLayer);

        const sketch = new Sketch({
            layer: graphicsLayer,
            view: view,
            availableCreateTools: ["polyline", "polygon"],
            creationMode: "update",
            updateOnGraphicClick: true,
            visibleElements: {
                createTools: {
                    point: false,
                    circle: false
                },
                selectionTools: {
                    "lasso-selection": false,
                    "rectangle-selection": false,
                },
                settingsMenu: false,
                undoRedoMenu: false
            }
        });
        view.ui.add(sketch, "top-right");

        // 结果图层
        const measurements = document.getElementById("measurements");
        view.ui.add(measurements, "manual");

        sketch.on("update", (e) => {
            const geometry = e.graphics[0].geometry;

            if (e.state === "start") {
                switchType(geometry);
            }

            if (e.state === "complete") {
                graphicsLayer.remove(graphicsLayer.graphics.getItemAt(0));
                measurements.innerHTML = null;
            }

            if (
                e.toolEventInfo &&
                (e.toolEventInfo.type === "scale-stop" ||
                    e.toolEventInfo.type === "reshape-stop" ||
                    e.toolEventInfo.type === "move-stop")
            ) {
                switchType(geometry);
            }

        });

        function getArea(polygon) {
            const geodesicArea = geometryEngine.geodesicArea(polygon, "square-meters");
            const planarArea = geometryEngine.planarArea(polygon, "square-meters");
            measurements.innerHTML =
                "<b>Geodesic area</b>:  " + geodesicArea.toFixed(2) + " m\xB2" + " |   <b>Planar area</b>: " + planarArea.toFixed(2) + "  m\xB2";
        }


        function getLength(line) {
            const geodesicLength = geometryEngine.geodesicLength(line, "meters");
            const planarLength = geometryEngine.planarLength(line, "meters");
            measurements.innerHTML =
                "<b>Geodesic length</b>:  " + geodesicLength.toFixed(2) + " m" + " |   <b>Planar length</b>: " + planarLength.toFixed(2) + "  m";
        }

        function switchType(geom) {
            switch (geom.type) {
                case "polygon":
                    getArea(geom);
                    break;
                case "polyline":
                    getLength(geom);
                    break;
                default:
                    console.log("No value found");
            }
        }
    });