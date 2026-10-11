import asyncio
import logging
import time
from typing import Any, Optional

import httpx

from app.config import settings
from app.models.location_context import EnvironmentType, LocationContext

logger = logging.getLogger(__name__)


class LocationContextService:
    def __init__(self):
        self._nominatim_lock = asyncio.Lock()
        self._last_nominatim_request = 0.0
        self._last_lat: float = 0.0
        self._last_lon: float = 0.0
        self._cache: dict[tuple[float, float], tuple[float, LocationContext]] = {}

    async def resolve(self, lat: Optional[float], lon: Optional[float]) -> LocationContext:
        if lat is None or lon is None:
            logger.info(
                "Location context: no coordinates provided; using general context"
            )
            return self._general_context(
                "A recent accurate location was unavailable; targets use general outdoor context."
            )

        # Use 4 decimal places (~11m) for cache key to avoid stale context for nearby movement
        coarse_lat = round(lat, 4)
        coarse_lon = round(lon, 4)
        cache_key = (coarse_lat, coarse_lon)
        cached = self._cache.get(cache_key)
        if cached and time.monotonic() - cached[0] < settings.GEOCONTEXT_CACHE_TTL_SECONDS:
            logger.info(
                "Location context: cache hit lat=%.4f lon=%.4f",
                coarse_lat,
                coarse_lon,
            )
            return cached[1].model_copy(deep=True)

        if settings.GEOCONTEXT_PROVIDER.casefold() != "openstreetmap":
            logger.warning(
                "Location context: provider not openstreetmap; using general context"
            )
            return self._general_context(
                "The configured geographical context provider is unavailable; targets use general outdoor context."
            )

        timeout = httpx.Timeout(settings.GEOCONTEXT_TIMEOUT_SECONDS)
        async with httpx.AsyncClient(
            timeout=timeout,
            headers={"User-Agent": settings.GEOCONTEXT_USER_AGENT},
        ) as client:
            reverse_data = await self._retry_lookup(
                "reverse-geocoding",
                lambda: self._reverse_geocode(client, coarse_lat, coarse_lon),
            )
            feature_data = await self._retry_lookup(
                "nearby-feature",
                lambda: self._nearby_features(client, coarse_lat, coarse_lon),
            )

        if reverse_data is None and feature_data is None:
            logger.warning(
                "Location context: both OSM lookups failed lat=%.4f lon=%.4f; using general context",
                coarse_lat,
                coarse_lon,
            )
            return self._general_context(
                "Nearby geographical data is unavailable; targets use general outdoor context."
            )

        address_value = (reverse_data or {}).get("address", {})
        address = address_value if isinstance(address_value, dict) else {}
        locality = next(
            (
                str(address[key])
                for key in ("city", "town", "village", "municipality", "hamlet", "suburb")
                if address.get(key)
            ),
            None,
        )
        region = address.get("state") or address.get("county")
        categories, feature_counts = self._feature_categories(feature_data)
        environment = self._environment_type(reverse_data, address, categories, feature_counts)
        has_both_sources = reverse_data is not None and feature_data is not None
        contextual = environment != "unknown" or bool(categories)
        context = LocationContext(
            locality=locality,
            region=region,
            environment_type=environment,
            nearby_categories=categories,
            nearby_feature_counts=feature_counts,
            context_mode="contextual" if contextual else "general",
            context_provider="openstreetmap",
            context_status="available" if has_both_sources else "partial",
            notice=(
                None
                if has_both_sources and contextual
                else "Only limited map context was available; targets use broad outdoor context."
            ),
        )
        logger.info(
            "Location context: resolved lat=%.4f lon=%.4f mode=%s status=%s env=%s locality=%s categories=%s counts=%s",
            coarse_lat,
            coarse_lon,
            context.context_mode,
            context.context_status,
            context.environment_type,
            context.locality,
            context.nearby_categories,
            context.nearby_feature_counts,
        )
        if context.context_status != "unavailable":
            if len(self._cache) >= 256:
                expired = [
                    key
                    for key, (cached_at, _) in self._cache.items()
                    if time.monotonic() - cached_at
                    >= settings.GEOCONTEXT_CACHE_TTL_SECONDS
                ]
                for key in expired:
                    self._cache.pop(key, None)
                if len(self._cache) >= 256:
                    oldest = min(self._cache, key=lambda key: self._cache[key][0])
                    self._cache.pop(oldest, None)
            self._cache[cache_key] = (time.monotonic(), context.model_copy(deep=True))
        return context

    async def _retry_lookup(self, name: str, lookup):
        for attempt in range(settings.GEOCONTEXT_RETRIES + 1):
            try:
                result = await lookup()
                if result is not None:
                    logger.info(
                        "Location context: %s lookup succeeded lat=%.4f lon=%.4f attempt=%d",
                        name,
                        self._last_lat if hasattr(self, '_last_lat') else 0,
                        self._last_lon if hasattr(self, '_last_lon') else 0,
                        attempt + 1,
                    )
                return result
            except (httpx.HTTPError, ValueError, TypeError) as exc:
                logger.warning(
                    "OpenStreetMap %s lookup failed lat=%.4f lon=%.4f attempt=%d error_type=%s",
                    name,
                    self._last_lat if hasattr(self, '_last_lat') else 0,
                    self._last_lon if hasattr(self, '_last_lon') else 0,
                    attempt + 1,
                    type(exc).__name__,
                )
                if attempt < settings.GEOCONTEXT_RETRIES:
                    await asyncio.sleep(0.25 * (attempt + 1))
        logger.error(
            "Location context: %s lookup exhausted retries lat=%.4f lon=%.4f",
            name,
            self._last_lat if hasattr(self, '_last_lat') else 0,
            self._last_lon if hasattr(self, '_last_lon') else 0,
        )
        return None

    async def _reverse_geocode(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> dict[str, Any]:
        self._last_lat = lat
        self._last_lon = lon
        async with self._nominatim_lock:
            wait_seconds = 1.0 - (time.monotonic() - self._last_nominatim_request)
            if wait_seconds > 0:
                await asyncio.sleep(wait_seconds)
            try:
                response = await client.get(
                    settings.GEOCONTEXT_NOMINATIM_URL,
                    params={
                        "format": "jsonv2",
                        "lat": lat,
                        "lon": lon,
                        "zoom": 14,
                        "addressdetails": 1,
                    },
                )
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ValueError("Reverse-geocoding response was not an object")
                return data
            finally:
                self._last_nominatim_request = time.monotonic()

    async def _nearby_features(
        self, client: httpx.AsyncClient, lat: float, lon: float
    ) -> dict[str, Any]:
        self._last_lat = lat
        self._last_lon = lon
        response = await client.post(
            settings.GEOCONTEXT_OVERPASS_URL,
            data={"data": self._overpass_query(lat, lon)},
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise ValueError("Nearby-feature response was not an object")
        if not isinstance(data.get("elements", []), list):
            raise ValueError("Nearby-feature elements were not a list")
        return data

    @staticmethod
    def _overpass_query(lat: float, lon: float) -> str:
        radius = settings.GEOCONTEXT_RADIUS_METERS
        return f"""
[out:json][timeout:7];
(
  nwr(around:{radius},{lat},{lon})[leisure~"^(park|playground|nature_reserve|sports_centre|pitch)$"];
  nwr(around:{radius},{lat},{lon})[natural~"^(wood|water|grassland)$"];
  nwr(around:{radius},{lat},{lon})[landuse~"^(residential|commercial|retail|grass|forest)$"];
  nwr(around:{radius},{lat},{lon})[highway~"^(residential|service|tertiary|secondary|primary|bus_stop|footway|pedestrian|path|cycleway)$"];
  nwr(around:{radius},{lat},{lon})[junction];
  nwr(around:{radius},{lat},{lon})[amenity~"^(bus_station|cafe|library|community_centre|bench|shelter)$"];
  nwr(around:{radius},{lat},{lon})[public_transport~"^(platform|station|stop_position)$"];
  nwr(around:{radius},{lat},{lon})[shop];
);
out tags center 80;
"""

    @staticmethod
    def _feature_categories(
        feature_data: Optional[dict[str, Any]],
    ) -> tuple[list[str], dict[str, int]]:
        counts: dict[str, int] = {}
        elements = (feature_data or {}).get("elements", [])
        for element in elements:
            if not isinstance(element, dict):
                continue
            tags = element.get("tags", {})
            if not isinstance(tags, dict):
                continue
            leisure = tags.get("leisure")
            natural = tags.get("natural")
            landuse = tags.get("landuse")
            highway = tags.get("highway")
            amenity = tags.get("amenity")
            public_transport = tags.get("public_transport")

            if leisure in {"park", "playground", "nature_reserve", "sports_centre", "pitch"}:
                counts["parks_and_recreation"] = counts.get("parks_and_recreation", 0) + 1
            if natural in {"wood", "water", "grassland"} or landuse in {"grass", "forest"}:
                counts["natural_areas"] = counts.get("natural_areas", 0) + 1
            if landuse == "residential":
                counts["residential_areas"] = counts.get("residential_areas", 0) + 1
            if landuse in {"commercial", "retail"} or tags.get("shop"):
                counts["commercial_areas"] = counts.get("commercial_areas", 0) + 1
            if highway in {
                "residential", "service", "tertiary", "secondary", "primary"
            } or tags.get("junction"):
                counts["roads_and_junctions"] = counts.get("roads_and_junctions", 0) + 1
            if highway in {"footway", "pedestrian", "path", "cycleway"}:
                counts["pedestrian_paths"] = counts.get("pedestrian_paths", 0) + 1
            if public_transport or highway == "bus_stop" or amenity == "bus_station":
                counts["public_transport"] = counts.get("public_transport", 0) + 1
            if amenity in {"cafe", "library", "community_centre", "bench", "shelter"}:
                counts["public_amenities"] = counts.get("public_amenities", 0) + 1

        return sorted(counts), counts

    @staticmethod
    def _environment_type(
        reverse_data: Optional[dict[str, Any]],
        address: dict[str, Any],
        categories: list[str],
        feature_counts: Optional[dict[str, int]] = None,
    ) -> EnvironmentType:
        feature_type = str((reverse_data or {}).get("type", "")).casefold()
        feature_class = str((reverse_data or {}).get("category", "")).casefold()
        if (
            feature_type in {"park", "forest", "nature_reserve", "wood"}
            or feature_class == "natural"
        ):
            return "park_nature"
        if address.get("hamlet") or address.get("farm") or address.get("isolated_dwelling"):
            return "rural"
        counts = feature_counts or {}
        if (
            address.get("city")
            or "public_transport" in categories
            or "commercial_areas" in categories
            or "roads_and_junctions" in categories
        ):
            return "urban"
        if "residential_areas" in categories or address.get("town") or address.get("suburb"):
            return "suburban"
        if counts.get("parks_and_recreation", 0) or counts.get("natural_areas", 0):
            return "park_nature"
        if address.get("village"):
            return "rural"
        return "unknown"

    @staticmethod
    def _general_context(notice: str) -> LocationContext:
        return LocationContext(context_mode="general", context_status="unavailable", notice=notice)


location_context_service = LocationContextService()
