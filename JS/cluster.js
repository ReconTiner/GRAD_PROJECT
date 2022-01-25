require([
    "esri/config",
    "esri/WebMap",
    "esri/views/MapView",
    "esri/Graphic",
    "esri/layers/CSVLayer",
    "esri/widgets/Legend",
    "esri/widgets/Expand",
    "esri/smartMapping/labels/clusters",
    "esri/smartMapping/popup/clusters",
    "esri/core/watchUtils",
    "esri/symbols/support/symbolUtils",
    "esri/geometry/geometryEngine"
],
    (esriConfig,
        Map,
        MapView,
        Graphic,
        CSVLayer,
        Legend,
        Expand,
        clusterLabelCreator,
        clusterPopupCreator,
        watchUtils,
        symbolUtils,
        geometryEngine) => {

        let layerView;

        esriConfig.apiKey = "AAPKf467f63bec014c8fa266ae623ee9f475ecFFIrG6H2k1U_gS-9Eygn9BggsmVW7pPxUaYfiUVSmMcTtFGjEvIqa59lIsHACP";

        const renderer = {
            type: "simple",
            symbol: {
                type: "simple-marker",
                style: "circle",
                color: [250, 250, 250],
                outline: {
                    color: [255, 255, 255, 0.5],
                    width: 0.5
                },
                size: "8px"
            },

            visualVariables: [
                {
                    type: "color",
                    field: "Analyst",
                    stops: [
                        { value: 0.03, color: "#2b83ba" },
                        { value: 0.08, color: "#abdda4" },
                        { value: 0.1, color: "#ffffbf" },
                        { value: 0.15, color: "#fdae61" },
                        { value: 0.2, color: "#d7191c" }
                    ]
                }
            ]

        };

        const layer = new CSVLayer({
            url: "../sample.csv",
            outFields: ["ImageName", "ImageDataTime", "Health", "Normal", "Badly", "Class"],
            title: "详细信息",
            popupTemplate: {
                title: "{ImageName}",
                content: [
                    {
                        type: "fields",
                        fieldInfos: [
                            {
                                fieldName: "Health"
                            },
                            {
                                fieldName: "Normal"
                            },
                            {
                                fieldName: "Badly"
                            }
                        ]
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
            },
            renderer: renderer
        });

        const map = new Map({
            basemap: "arcgis-imagery",
            layers: [layer]
        });

        const view = new MapView({
            container: "main-box",
            map,
            center: [117.5526377222, 34.3100582222],
            zoom: 18,
        });

        // 重写选中聚类要素的外观
        view.popup.viewModel.selectedClusterBoundaryFeature.symbol = {
            type: "simple-fill",
            style: "solid",
            color: "rgba(50,50,50,0.15)",
            outline: {
                width: 0.5,
                color: "rgba(50,50,50,0.25)"
            }
        };

        const legend = new Legend({
            view,
            container: "legendDiv"
        });

        const infoDiv = document.getElementById("infoDiv");
        view.ui.add(
            new Expand({
                view,
                content: infoDiv,
                expandIconClass: "esri-icon-layer-list",
                expanded: true
            }),
            "top-left"
        );

        layer
            .when()
            .then(generateClusterConfig)
            .then(async (featureReduction) => {
                layer.featureReduction = featureReduction;
                layerView = await view.whenLayerView(layer);
                const toggleButton = document.getElementById("toggle-cluster");
                toggleButton.addEventListener("click", toggleClustering);

                // To turn off clustering on a layer, set the
                // featureReduction property to null
                function toggleClustering() {
                    if (isWithinScaleThreshold()) {
                        let fr = layer.featureReduction;
                        layer.featureReduction =
                            fr && fr.type === "cluster" ? null : featureReduction;
                    }
                    toggleButton.innerText =
                        toggleButton.innerText === "Enable Clustering"
                            ? "Disable Clustering"
                            : "Enable Clustering";
                }

                view.whenLayerView(layer).then((layerView) => {
                    const filterSelect = document.getElementById("filter");
                    // filters the layer using a definitionExpression
                    filterSelect.addEventListener("change", (event) => {
                        const newValue = event.target.value;

                        const whereClause = newValue
                            ? `Class = '${newValue}'`
                            : null;
                        layerView.filter = {
                            where: whereClause
                        };
                        // close popup for former cluster that no longer displays
                        view.popup.close();
                    });
                });

                view.watch("scale", (scale) => {
                    if (toggleButton.innerText === "Disable Clustering") {
                        layer.featureReduction = isWithinScaleThreshold()
                            ? featureReduction
                            : null;
                    }
                });
            })
            .catch((error) => {
                console.error(error);
            });

        function isWithinScaleThreshold() {
            return view.scale > 50;
        }

        async function generateClusterConfig(layer) {
            // 默认弹窗
            const popupTemplate = await clusterPopupCreator
                .getTemplates({ layer })
                .then(
                    (popupTemplateResponse) => popupTemplateResponse.primaryTemplate.value
                );
            popupTemplate.actions = [
                {
                    title: "Statistics",
                    id: "statistics",
                    className: "esri-icon-line-chart"
                },
                {
                    title: "Convex hull",
                    id: "convex-hull",
                    className: "esri-icon-polygon"
                },
                {
                    title: "Show features",
                    id: "show-features",
                    className: "esri-icon-maps"
                }
            ];

            // 默认标签
            const { labelingInfo, clusterMinSize } = await clusterLabelCreator
                .getLabelSchemes({ layer, view })
                .then((labelSchemes) => labelSchemes.primaryScheme);

            return {
                type: "cluster",
                popupTemplate,
                labelingInfo,
                clusterMinSize
            };
        }

        // 为弹窗添加事件监控
        view.popup.on("trigger-action", (event) => {
            clearViewGraphics();

            const popup = view.popup;
            const selectedFeature = popup.selectedFeature && popup.selectedFeature.isAggregate;

            const id = event.action.id;

            if (id === "convex-hull") {
                displayConvexHull(view.popup.selectedFeature);
            }
            if (id === "show-features") {
                displayFeatures(view.popup.selectedFeature);
            }
            if (id === "statistics") {
                calculateStatistics(view.popup.selectedFeature);
            }
        });
        watchUtils.watch(
            view,
            ["scale", "popup.selectedFeature", "popup.visible"],
            clearViewGraphics
        );

        let convexHullGraphic = null;
        let clusterChildGraphics = [];

        function clearViewGraphics() {
            view.graphics.remove(convexHullGraphic);
            view.graphics.removeMany(clusterChildGraphics);
        }

        // 显示选中聚类中的所有要素
        async function displayFeatures(graphic) {
            processParams(graphic, layerView);

            const query = layerView.createQuery();
            query.aggregateIds = [graphic.getObjectId()];
            const { features } = await layerView.queryFeatures(query);

            features.forEach(async (feature) => {
                const symbol = await symbolUtils.getDisplayedSymbol(feature);
                feature.symbol = symbol;
                view.graphics.add(feature);
            });
            clusterChildGraphics = features;
        }

        // 显示选中聚类元素的外接矩形
        async function displayConvexHull(graphic) {
            processParams(graphic, layerView);

            const query = layerView.createQuery();
            query.aggregateIds = [graphic.getObjectId()];
            const { features } = await layerView.queryFeatures(query);
            const geometries = features.map((feature) => feature.geometry);
            const [convexHull] = geometryEngine.convexHull(geometries, true);

            convexHullGraphic = new Graphic({
                geometry: convexHull,
                symbol: {
                    type: "simple-fill",
                    outline: {
                        width: 1.5,
                        color: [75, 75, 75, 1]
                    },
                    style: "none",
                    color: [0, 0, 0, 0.1]
                }
            });
            view.graphics.add(convexHullGraphic);
        }

        // 显示统计信息
        async function calculateStatistics(graphic) {
            processParams(graphic, layerView);

            const query = layerView.createQuery();

            query.aggregateIds = [graphic.getObjectId()];

            // query.groupByFieldsForStatistics = ["fuel1"];
            query.groupByFieldsForStatistics = ["Class"];
            // query.outFields = ["capacity_mw", "fuel1"];
            query.outFields = ["Health", "Normal", "Badly", "Class"];
            // query.orderByFields = ["num_features desc"];
            query.orderByFields = ["badly_total desc"];

            query.outStatistics = [
                {
                    onStatisticField: "Badly",
                    outStatisticFieldName: "badly_total",
                    statisticType: "sum"
                },
                {
                    onStatisticField: "Health",
                    outStatisticFieldName: "health_total",
                    statisticType: "sum"
                },
                {
                    onStatisticField: "1",
                    outStatisticFieldName: "num_features",
                    statisticType: "count"
                },
                // {
                //     onStatisticField: "capacity_mw",
                //     outStatisticFieldName: "capacity_max",
                //     statisticType: "max"
                // }
            ];

            const { features } = await layerView.queryFeatures(query);
            const stats = features.map((feature) => feature.attributes);

            let table = `
          <div class="table-container">
            <span style="font-size: 14pt"><strong>Summary by class type</strong></span>
            <br/>
            <br/>
            <table class="esri-widget popup">
              <tr class="head"><td>Class</td><td>Count</td><td>Health Total</td><td>Badly Total</td></tr>
        `;

            let totalBadly = 0;
            let totalCount = 0;

            stats.forEach((stat) => {
                const cls = stat.Class;
                const count = stat.num_features;
                const health = stat.health_total;
                const badly = stat.badly_total;


                totalBadly += badly;
                totalCount += count;

                table += `
            <tr><td><span style:'font-weight:bolder'>${cls}</span></td><td class="num">${count}</td><td class="num">${roundDecimals(
                    health,
                    2
                ).toLocaleString()}</td><td class="num">${roundDecimals(
                    badly,
                    2
                ).toLocaleString()}</td></tr>
          `;
            });

            table += `
          </table>
        </div>`;

            view.popup.content =
                `
          <div style="font-size: 12pt">
          Number of features: <strong>${totalCount.toLocaleString()}</strong><br>
          Total Badly: <strong>${roundDecimals(
                    totalBadly,
                    2
                ).toLocaleString()}</strong><br>
          </div><br>
        ` + table;
        }

        function processParams(graphic, layerView) {
            if (!graphic || !layerView) {
                throw new Error("Graphic or layerView not provided.");
            }

            if (!graphic.isAggregate) {
                throw new Error("Graphic must represent a cluster.");
            }
        }

        function roundDecimals(num, places) {
            return Math.round(num * Math.pow(10, places)) / Math.pow(10, places);
        }
    });