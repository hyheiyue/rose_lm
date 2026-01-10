/**
 * This file is part of Small Point-LIO, an advanced Point-LIO algorithm implementation.
 * Copyright (C) 2025  Yingjie Huang
 * Licensed under the MIT License. See License.txt in the project root for license information.
 */

#pragma once

#include <Eigen/Dense>
#include <ankerl/unordered_dense.h>
#include <list>
namespace rose_lm {
namespace small_point_lio {
    inline uint64_t hash_position_index(const Eigen::Matrix<int16_t, 3, 1>& v);

    class PointWithDistance {
    public:
        Eigen::Vector3f point;
        float distance = 0;

        PointWithDistance(Eigen::Vector3f point, float distance);
        PointWithDistance(): point(Eigen::Vector3f::Zero()), distance(0.0f) {}

        bool operator()(const PointWithDistance& p1, const PointWithDistance& p2) const;

        bool operator<(const PointWithDistance& rhs) const;
    };

    class SmallIVox {
    private:
        ankerl::unordered_dense::map<uint64_t, std::list<Eigen::Vector3f>::iterator> grids_map;
        float inv_resolution;
        size_t capacity;
        std::list<Eigen::Vector3f> grids_cache_;
        std::vector<PointWithDistance> candidates;

    public:
        explicit SmallIVox(float resolution, size_t capacity);

        bool add_point(const Eigen::Vector3f& point_to_add);

        void get_closest_point(
            const Eigen::Vector3f& pt,
            std::vector<Eigen::Vector3f>& closest_pt,
            size_t max_num = 5
        );
        inline void get_closest_points_batch(
            const std::vector<Eigen::Vector3f>& pts,
            std::vector<std::vector<Eigen::Vector3f>>& closest_pts,
            size_t max_num
        );
        inline __attribute__((always_inline)) Eigen::Vector3i unpack_position_index(uint64_t key) {
            const int x = static_cast<int16_t>(key & 0xFFFFULL);
            const int y = static_cast<int16_t>((key >> 16) & 0xFFFFULL);
            const int z = static_cast<int16_t>((key >> 32) & 0xFFFFULL);
            return { x, y, z };
        }
        inline __attribute__((always_inline)) uint64_t
        pack_position_index(const Eigen::Vector3i& idx) {
            return (static_cast<uint64_t>(static_cast<uint16_t>(idx.z())) << 32)
                | (static_cast<uint64_t>(static_cast<uint16_t>(idx.y())) << 16)
                | (static_cast<uint64_t>(static_cast<uint16_t>(idx.x())));
        }

        [[nodiscard]] Eigen::Matrix<uint16_t, 3, 1> get_position_index(const Eigen::Vector3f& pt
        ) const;
        // const std::vector<Eigen::Vector3i> neighbor_offs = {
        //     // 6 face neighbors
        //     { 0, 0, 0 },
        //     { 1, 0, 0 },
        //     { -1, 0, 0 },
        //     { 0, 1, 0 },
        //     { 0, -1, 0 },
        //     { 0, 0, 1 },
        //     { 0, 0, -1 },

        //     // 12 edge neighbors
        //     { 1, 1, 0 },
        //     { 1, -1, 0 },
        //     { -1, 1, 0 },
        //     { -1, -1, 0 },
        //     { 1, 0, 1 },
        //     { 1, 0, -1 },
        //     { -1, 0, 1 },
        //     { -1, 0, -1 },
        //     { 0, 1, 1 },
        //     { 0, 1, -1 },
        //     { 0, -1, 1 },
        //     { 0, -1, -1 }
        // };
        const std::vector<Eigen::Vector3i> neighbor_offs = { { 0, 0, 0 },  { 1, 0, 0 },
                                                             { -1, 0, 0 }, { 0, 1, 0 },
                                                             { 0, -1, 0 }, { 0, 0, 1 },
                                                             { 0, 0, -1 } };
    };

    inline __attribute__((always_inline)) uint64_t
    hash_position_index(const Eigen::Matrix<uint16_t, 3, 1>& v) {
        return (static_cast<uint64_t>(v[0]) << 32) | (static_cast<uint64_t>(v[1]) << 16)
            | static_cast<uint64_t>(v[2]);
    }

    inline __attribute__((always_inline))
    PointWithDistance::PointWithDistance(Eigen::Vector3f point, float distance):
        point(std::move(point)),
        distance(distance) {}

    inline __attribute__((always_inline)) bool
    PointWithDistance::operator()(const PointWithDistance& p1, const PointWithDistance& p2) const {
        return p1.distance < p2.distance;
    }

    inline __attribute__((always_inline)) bool
    PointWithDistance::operator<(const PointWithDistance& rhs) const {
        return distance < rhs.distance;
    }

    inline __attribute__((always_inline)) SmallIVox::SmallIVox(float resolution, size_t capacity):
        inv_resolution(1 / resolution),
        capacity(capacity) {}

    inline __attribute__((always_inline)) void SmallIVox::get_closest_point(
        const Eigen::Vector3f& pt,
        std::vector<Eigen::Vector3f>& closest_pt,
        size_t max_num
    ) {
        closest_pt.clear();
        Eigen::Matrix<uint16_t, 3, 1> key = get_position_index(pt);
        uint64_t hash_key = hash_position_index(key);

        const Eigen::Vector3i base_idx = unpack_position_index(hash_key);

        std::vector<Eigen::Vector3f> neighbor_pts;
        neighbor_pts.reserve(neighbor_offs.size());

        for (const auto& off: neighbor_offs) {
            Eigen::Vector3i nidx = base_idx + off;
            uint64_t nkey = pack_position_index(nidx);
            auto iter = grids_map.find(nkey);
            if (iter != grids_map.end())
                neighbor_pts.push_back(*iter->second);
        }

        if (neighbor_pts.empty()) [[unlikely]] {
            return;
        }
        if (neighbor_pts.size() > max_num) [[likely]] {
            std::vector<PointWithDistance> candidates;
            candidates.reserve(neighbor_pts.size());
            for (auto& p: neighbor_pts)
                candidates.emplace_back(p, (p - pt).squaredNorm());

            std::nth_element(
                candidates.begin(),
                candidates.begin() + static_cast<std::ptrdiff_t>(max_num) - 1,
                candidates.end()
            );

            closest_pt.reserve(max_num);
            for (size_t i = 0; i < max_num; ++i)
                closest_pt.push_back(candidates[i].point);
        } else {
            closest_pt = std::move(neighbor_pts);
        }
    }
    inline __attribute__((always_inline)) void SmallIVox::get_closest_points_batch(
        const std::vector<Eigen::Vector3f>& pts,
        std::vector<std::vector<Eigen::Vector3f>>& closest_pts,
        size_t max_num
    ) {
        closest_pts.resize(pts.size());

        std::unordered_map<uint64_t, std::vector<size_t>> voxel_points;
        voxel_points.reserve(pts.size());
        for (size_t i = 0; i < pts.size(); ++i) {
            uint64_t key = hash_position_index(get_position_index(pts[i]));
            voxel_points[key].push_back(i);
        }
        for (const auto& [hash_key, indices]: voxel_points) {
            const Eigen::Vector3i base_idx = unpack_position_index(hash_key);

            std::vector<Eigen::Vector3f> neighbor_pts;
            neighbor_pts.reserve(8);

            for (const auto& off: neighbor_offs) {
                Eigen::Vector3i idx = base_idx + off;
                uint64_t nkey = pack_position_index(idx);

                auto iter = grids_map.find(nkey);
                if (iter != grids_map.end())
                    neighbor_pts.push_back(*iter->second);
            }

            for (auto idx: indices) {
                const auto& pt = pts[idx];

                if (neighbor_pts.empty()) {
                    closest_pts[idx].clear();
                    continue;
                }

                std::vector<PointWithDistance> candidates;
                candidates.reserve(neighbor_pts.size());

                for (auto& p: neighbor_pts)
                    candidates.emplace_back(p, (p - pt).squaredNorm());

                if (candidates.size() > max_num) {
                    std::nth_element(
                        candidates.begin(),
                        candidates.begin() + static_cast<std::ptrdiff_t>(max_num) - 1,
                        candidates.end()
                    );
                    candidates.resize(max_num);
                }

                closest_pts[idx].resize(candidates.size());
                for (size_t k = 0; k < candidates.size(); ++k)
                    closest_pts[idx][k] = candidates[k].point;
            }
        }
    }
    inline __attribute__((always_inline)) bool SmallIVox::add_point(const Eigen::Vector3f& point) {
        Eigen::Matrix<uint16_t, 3, 1> key = get_position_index(point);
        auto hash_key = hash_position_index(key);
        auto iter = grids_map.find(hash_key);
        if (iter != grids_map.end()) {
            grids_cache_.splice(grids_cache_.begin(), grids_cache_, iter->second);
            grids_map[hash_key] = grids_cache_.begin();
            return false;
        } else {
            grids_cache_.push_front(point);
            grids_map.emplace(hash_key, grids_cache_.begin());
            if (grids_map.size() >= capacity) {
                grids_map.erase(hash_position_index(get_position_index(grids_cache_.back())));
                grids_cache_.pop_back();
            }
            return true;
        }
    }

    [[nodiscard]] inline __attribute__((always_inline)) Eigen::Matrix<uint16_t, 3, 1>
    SmallIVox::get_position_index(const Eigen::Vector3f& pt) const {
        return (pt * inv_resolution).array().floor().cast<uint16_t>();
    }

} // namespace small_point_lio

} // namespace rose_lm
